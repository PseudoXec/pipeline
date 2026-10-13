"""SQLite-backed asynchronous buffer for vehicle and person detection events."""
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from api.result import SendResult

log = logging.getLogger("pipeline")

_SENDER_BATCH_ROWS = 20
_MAX_BACKOFF_SECONDS = 60          # a failed row is retried within a minute once the API is back
_INITIAL_BACKOFF_SECONDS = 5
_MAX_CONSECUTIVE_FAILURES = 3      # API looks down: stop this pass instead of timing out on every row
_REJECT_RETRY_SECONDS = 30


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode an OpenCV BGR image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to JPEG-encode image")
    return buffer.tobytes()


@dataclass
class VehicleRecord:
    track_id: str
    camera_source: str
    vehicle_class: str
    vehicle_confidence: float
    vehicle_image_jpeg: Optional[bytes] = None
    vehicle_box_x1: Optional[float] = None
    vehicle_box_y1: Optional[float] = None
    vehicle_box_x2: Optional[float] = None
    vehicle_box_y2: Optional[float] = None
    plate_detected: bool = False
    plate_confidence: Optional[float] = None
    plate_image_jpeg: Optional[bytes] = None
    plate_box_x1: Optional[float] = None
    plate_box_y1: Optional[float] = None
    plate_box_x2: Optional[float] = None
    plate_box_y2: Optional[float] = None
    event_uuid: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    disk_files: Optional[Dict[str, bytes]] = None


@dataclass
class PersonRecord:
    event_uuid: str
    device_id: str
    session_id: str
    track_id: str
    person_confidence: float
    person_image_jpeg: bytes
    face_detected: bool
    detected_at: datetime = field(default_factory=datetime.now)
    face_image_jpeg: Optional[bytes] = None
    face_landmarks: Optional[str] = None
    face_quality_score: Optional[float] = None
    disk_files: Optional[Dict[str, bytes]] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id            TEXT NOT NULL,
    camera_source       TEXT NOT NULL,
    vehicle_class       TEXT NOT NULL,
    vehicle_confidence  REAL NOT NULL,
    vehicle_image       BLOB,
    vehicle_box_x1      REAL,
    vehicle_box_y1      REAL,
    vehicle_box_x2      REAL,
    vehicle_box_y2      REAL,
    plate_detected      INTEGER NOT NULL,
    plate_confidence    REAL,
    plate_image         BLOB,
    plate_box_x1        REAL,
    plate_box_y1        REAL,
    plate_box_x2        REAL,
    plate_box_y2        REAL,
    detected_at         TEXT NOT NULL,
    camera_name         TEXT,
    camera_ip           TEXT,
    camera_location     TEXT,
    event_uuid          TEXT,
    synced              INTEGER NOT NULL DEFAULT 0,
    sync_attempts       INTEGER NOT NULL DEFAULT 0,
    reject_attempts     INTEGER NOT NULL DEFAULT 0,
    last_sync_attempt_at TEXT,
    sync_error          TEXT,
    disk_files          TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_detections_synced ON detections (synced, id);
CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections (detected_at);

CREATE TABLE IF NOT EXISTS person_detections (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uuid           TEXT NOT NULL UNIQUE,
    device_id            TEXT NOT NULL,
    session_id           TEXT NOT NULL,
    track_id             TEXT NOT NULL,
    camera_name          TEXT,
    camera_ip            TEXT,
    camera_location      TEXT,
    person_confidence    REAL NOT NULL,
    person_image         BLOB NOT NULL,
    face_detected        INTEGER NOT NULL CHECK (face_detected IN (0, 1)),
    face_image           BLOB,
    face_landmarks       TEXT,
    face_quality_score   REAL,
    detected_at          TEXT NOT NULL,
    synced               INTEGER NOT NULL DEFAULT 0,
    sync_attempts        INTEGER NOT NULL DEFAULT 0,
    reject_attempts      INTEGER NOT NULL DEFAULT 0,
    last_sync_attempt_at TEXT,
    synced_at            TEXT,
    sync_error           TEXT,
    disk_files           TEXT,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_person_synced ON person_detections (synced, id);
CREATE INDEX IF NOT EXISTS idx_person_detected_at ON person_detections (detected_at);
"""


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def _ts(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat(sep=" ", timespec="seconds") if value is not None else None


class SQLiteBuffer:
    """A batching SQLite writer and an independent HTTP sender, each with its own connection."""

    def __init__(self, config, camera_info=None, api_sender=None):
        self.config = config
        self.storage_cfg = config.storage
        self.database_path = self.storage_cfg.database_path
        self.output_dir = self.storage_cfg.output_dir
        self.batch_size = getattr(self.storage_cfg, "write_batch_size", 8)
        self.flush_interval_seconds = getattr(self.storage_cfg, "write_flush_interval_seconds", 1.0)
        self.delete_disk_images_after_send = self.storage_cfg.delete_disk_images_after_send
        self.retention_days = self.storage_cfg.retention_days
        self.max_unsynced_days = self.storage_cfg.max_unsynced_days
        self.max_reject_attempts = getattr(self.storage_cfg, "max_reject_attempts", 3)
        self.camera_info = camera_info
        self.api_sender = api_sender
        self.camera_name = getattr(camera_info, "name", None) if camera_info else None
        self.camera_ip = getattr(camera_info, "ip_address", None) if camera_info else None
        self.camera_location = getattr(camera_info, "location", None) if camera_info else None

        os.makedirs(os.path.dirname(os.path.abspath(self.database_path)) or ".", exist_ok=True)
        self._queue: "queue.Queue[Union[VehicleRecord, PersonRecord]]" = queue.Queue()
        self._disk_queue: "queue.Queue[List[Tuple[str, int, Union[VehicleRecord, PersonRecord]]]]" = queue.Queue()
        self._disk_pending_ids: Set[Tuple[str, int]] = set()
        self._disk_pending_lock = threading.Lock()
        self._writer_stop = threading.Event()
        self._disk_stop = threading.Event()
        self._sender_stop = threading.Event()
        self._sender_wake = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._disk_thread: Optional[threading.Thread] = None
        self._sender_thread: Optional[threading.Thread] = None

        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
            self._migrate_schema(connection)
            self._cleanup_disallowed_rows(connection)
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        additions = {
            "detections": {
                "sync_attempts": "INTEGER NOT NULL DEFAULT 0",
                "reject_attempts": "INTEGER NOT NULL DEFAULT 0",
                "last_sync_attempt_at": "TEXT",
                "sync_error": "TEXT",
                "disk_files": "TEXT",
                "event_uuid": "TEXT",
            },
            "person_detections": {
                "reject_attempts": "INTEGER NOT NULL DEFAULT 0",
                "disk_files": "TEXT",
            },
        }
        for table, columns in additions.items():
            existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
        connection.commit()

    def _cleanup_disallowed_rows(self, connection: sqlite3.Connection) -> None:
        vehicle_cfg = self.config.vehicle.features
        if vehicle_cfg.plate_detection and not vehicle_cfg.store_without_plate:
            with connection:
                deleted = connection.execute(
                    "DELETE FROM detections WHERE plate_detected = 0",
                ).rowcount
            if deleted:
                log.info("[storage] removed %d existing vehicle row(s) without plates", deleted)

        person_cfg = self.config.person
        if person_cfg.face_enabled and not person_cfg.store_without_face:
            with connection:
                deleted = connection.execute(
                    "DELETE FROM person_detections WHERE face_detected = 0",
                ).rowcount
            if deleted:
                log.info("[storage] removed %d existing person row(s) without faces", deleted)

    def start(self) -> "SQLiteBuffer":
        self._sender_thread = threading.Thread(target=self._sender_loop, name="sqlite-sender", daemon=True)
        self._writer_thread = threading.Thread(target=self._writer_loop, name="sqlite-buffer", daemon=True)
        self._disk_thread = threading.Thread(target=self._disk_writer_loop, name="image-file-writer", daemon=True)
        self._sender_thread.start()
        self._writer_thread.start()
        self._disk_thread.start()
        self._sender_wake.set()
        log.info("[storage] SQLite writer and sender started (%s)", self.database_path)
        return self

    def enqueue(self, record: Union[VehicleRecord, PersonRecord]) -> None:
        self._queue.put(record)

    def stop(self) -> None:
        self._writer_stop.set()
        self._sender_stop.set()
        self._sender_wake.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10.0)
        self._disk_stop.set()
        if self._disk_thread is not None:
            self._disk_thread.join(timeout=5.0)
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=1.0)
        log.info("[storage] SQLite writer and sender stopped")

    def _writer_loop(self) -> None:
        connection = self._connect()
        last_flush = time.monotonic()
        last_retention = 0.0
        write_failures = 0
        pending: List[Union[VehicleRecord, PersonRecord]] = []
        try:
            while not self._writer_stop.is_set() or not self._queue.empty() or pending:
                try:
                    pending.append(self._queue.get(timeout=0.2))
                except queue.Empty:
                    pass

                now = time.monotonic()
                flush = (len(pending) >= self.batch_size or
                         (pending and now - last_flush >= self.flush_interval_seconds))
                if flush:
                    try:
                        self._write_batch(connection, pending)
                    except sqlite3.Error:
                        write_failures += 1
                        log.exception("[storage] batch write failed (%d/5)", write_failures)
                        if write_failures < 5:
                            last_flush = now          # keep the records and try again on the next flush
                            time.sleep(0.5)
                            continue
                        log.error("[storage] dropping %d record(s) after repeated write failures", len(pending))
                    write_failures = 0
                    pending.clear()
                    last_flush = now
                    self._sender_wake.set()

                if now - last_retention >= 6 * 3600:
                    self._run_retention(connection)
                    last_retention = now
        except Exception:
            log.exception("[storage] writer thread failed")
            raise
        finally:
            connection.close()

    def _write_batch(
        self, connection: sqlite3.Connection, records: List[Union[VehicleRecord, PersonRecord]],
    ) -> None:
        disk_tasks: List[Tuple[str, int, Union[VehicleRecord, PersonRecord]]] = []
        try:
            with connection:
                cursor = connection.cursor()
                for record in records:
                    disk_names = json.dumps(list(record.disk_files or {}))
                    if isinstance(record, VehicleRecord):
                        cursor.execute(
                            """
                            INSERT INTO detections (
                                track_id, camera_source, vehicle_class, vehicle_confidence, vehicle_image,
                                vehicle_box_x1, vehicle_box_y1, vehicle_box_x2, vehicle_box_y2,
                                plate_detected, plate_confidence, plate_image,
                                plate_box_x1, plate_box_y1, plate_box_x2, plate_box_y2,
                                detected_at, camera_name, camera_ip, camera_location, disk_files, event_uuid
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                record.track_id, record.camera_source, record.vehicle_class,
                                record.vehicle_confidence, record.vehicle_image_jpeg,
                                record.vehicle_box_x1, record.vehicle_box_y1, record.vehicle_box_x2,
                                record.vehicle_box_y2, int(record.plate_detected), record.plate_confidence,
                                record.plate_image_jpeg, record.plate_box_x1, record.plate_box_y1,
                                record.plate_box_x2, record.plate_box_y2, _ts(record.detected_at),
                                self.camera_name, self.camera_ip, self.camera_location, disk_names, record.event_uuid,
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO person_detections (
                                event_uuid, device_id, session_id, track_id, camera_name, camera_ip, camera_location,
                                person_confidence, person_image, face_detected, face_image, face_landmarks,
                                face_quality_score, detected_at, disk_files
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                record.event_uuid, record.device_id, record.session_id, record.track_id,
                                self.camera_name, self.camera_ip, self.camera_location, record.person_confidence,
                                record.person_image_jpeg, int(record.face_detected), record.face_image_jpeg,
                                record.face_landmarks, record.face_quality_score, _ts(record.detected_at),
                                disk_names,
                            ),
                        )
                    if record.disk_files:
                        task = ("detections" if isinstance(record, VehicleRecord) else "person_detections",
                                cursor.lastrowid, record)
                        with self._disk_pending_lock:
                            self._disk_pending_ids.add((task[0], task[1]))
                        disk_tasks.append(task)
        except Exception:
            with self._disk_pending_lock:
                for table, row_id, _record in disk_tasks:
                    self._disk_pending_ids.discard((table, row_id))
            raise
        if disk_tasks:
            self._disk_queue.put(disk_tasks)

    def _disk_writer_loop(self) -> None:
        while not self._disk_stop.is_set() or not self._disk_queue.empty():
            try:
                records = self._disk_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            for table, row_id, record in records:
                self._write_disk_files(record)
                with self._disk_pending_lock:
                    self._disk_pending_ids.discard((table, row_id))
            self._sender_wake.set()

    def _sender_loop(self) -> None:
        connection = self._connect()
        try:
            while not self._sender_stop.is_set():
                try:
                    if self.api_sender is not None and self.config.switches.api:
                        self._send_unsynced(connection, "detections")
                        if not self._sender_stop.is_set():
                            self._send_unsynced(connection, "person_detections")
                except Exception:
                    log.exception("[storage] sender pass failed")
                self._sender_wake.wait(timeout=1.0)
                self._sender_wake.clear()
        finally:
            connection.close()

    def _send_unsynced(self, connection: sqlite3.Connection, table: str) -> None:
        if table == "detections" and not self.config.vehicle.features.send_via_api:
            return
        if table == "person_detections" and not self.config.person.send_via_api:
            return

        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE synced = 0 ORDER BY id LIMIT ?",
                (_SENDER_BATCH_ROWS * 4,),
            ).fetchall()
        finally:
            connection.row_factory = None

        sent = 0
        attempted = 0
        consecutive_failures = 0
        for row in rows:
            if self._sender_stop.is_set():
                break
            with self._disk_pending_lock:
                if (table, row["id"]) in self._disk_pending_ids:
                    continue
            if not self._retry_due(row):
                continue
            if attempted >= _SENDER_BATCH_ROWS:
                break
            attempted += 1
            record = self._vehicle_from_row(row) if table == "detections" else self._person_from_row(row)
            outcome = self.api_sender.send(record)
            result, detail = outcome
            if result == SendResult.SENT:
                self._mark_sent(connection, table, row["id"], row["disk_files"])
                sent += 1
                consecutive_failures = 0
            elif result == SendResult.REJECTED:
                self._record_rejection(connection, table, row, detail)
                consecutive_failures = 0
            else:
                self._record_failure(connection, table, row["id"], detail or "send failed")
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    break
        if sent:
            log.info("[storage] sent %d %s event(s)", sent, "vehicle" if table == "detections" else "person")

    @staticmethod
    def _retry_due(row: sqlite3.Row) -> bool:
        last_attempt = row["last_sync_attempt_at"]
        if not last_attempt:
            return True
        try:
            attempted_at = datetime.fromisoformat(last_attempt)
        except ValueError:
            log.warning("[storage] invalid last_sync_attempt_at for row %s: %r", row["id"], last_attempt)
            return True
        rejections = int(row["reject_attempts"])
        if rejections > 0:
            delay = _REJECT_RETRY_SECONDS * rejections
        else:
            attempts = max(1, int(row["sync_attempts"]))
            delay = min(_MAX_BACKOFF_SECONDS, _INITIAL_BACKOFF_SECONDS * (2 ** min(attempts - 1, 10)))
        return (datetime.now() - attempted_at).total_seconds() >= delay

    @staticmethod
    def _disk_paths(value: Optional[str]) -> List[str]:
        if not value:
            return []
        try:
            paths = json.loads(value)
        except (TypeError, ValueError) as error:
            log.warning("[storage] invalid persisted disk file list: %s", error)
            return []
        return [path for path in paths if isinstance(path, str)]

    def _vehicle_from_row(self, row: sqlite3.Row) -> VehicleRecord:
        return VehicleRecord(
            track_id=row["track_id"], camera_source=row["camera_source"], vehicle_class=row["vehicle_class"],
            vehicle_confidence=row["vehicle_confidence"], vehicle_image_jpeg=row["vehicle_image"],
            vehicle_box_x1=row["vehicle_box_x1"], vehicle_box_y1=row["vehicle_box_y1"],
            vehicle_box_x2=row["vehicle_box_x2"], vehicle_box_y2=row["vehicle_box_y2"],
            plate_detected=bool(row["plate_detected"]), plate_confidence=row["plate_confidence"],
            plate_image_jpeg=row["plate_image"], plate_box_x1=row["plate_box_x1"],
            plate_box_y1=row["plate_box_y1"], plate_box_x2=row["plate_box_x2"],
            plate_box_y2=row["plate_box_y2"], detected_at=datetime.fromisoformat(row["detected_at"]),
            event_uuid=row["event_uuid"],
            disk_files={path: b"" for path in self._disk_paths(row["disk_files"])},
        )

    def _person_from_row(self, row: sqlite3.Row) -> PersonRecord:
        return PersonRecord(
            event_uuid=row["event_uuid"], device_id=row["device_id"], session_id=row["session_id"],
            track_id=row["track_id"], person_confidence=row["person_confidence"],
            person_image_jpeg=row["person_image"], face_detected=bool(row["face_detected"]),
            detected_at=datetime.fromisoformat(row["detected_at"]), face_image_jpeg=row["face_image"],
            face_landmarks=row["face_landmarks"], face_quality_score=row["face_quality_score"],
            disk_files={path: b"" for path in self._disk_paths(row["disk_files"])},
        )

    @staticmethod
    def _record_failure(connection: sqlite3.Connection, table: str, row_id: int, detail: str) -> None:
        with connection:
            connection.execute(
                f"UPDATE {table} SET sync_attempts = sync_attempts + 1, last_sync_attempt_at = ?, "
                "sync_error = ?, synced = 0 WHERE id = ?",
                (_now(), detail[:300], row_id),
            )

    def _record_rejection(
        self, connection: sqlite3.Connection, table: str, row: sqlite3.Row, detail: Optional[str],
    ) -> None:
        attempts = int(row["reject_attempts"]) + 1
        error = detail or "permanent HTTP rejection"
        if attempts >= max(1, self.max_reject_attempts):
            with connection:
                connection.execute(f"DELETE FROM {table} WHERE id = ?", (row["id"],))
            self._delete_disk_files(self._disk_paths(row["disk_files"]))
            log.warning("[storage] dropping rejected %s row %s after %d permanent rejections: %s",
                        table, row["id"], attempts, error)
            return
        with connection:
            connection.execute(
                f"UPDATE {table} SET sync_attempts = sync_attempts + 1, reject_attempts = ?, "
                "last_sync_attempt_at = ?, sync_error = ?, synced = 0 "
                "WHERE id = ?",
                (attempts, _now(), error[:300], row["id"]),
            )

    def _mark_sent(self, connection: sqlite3.Connection, table: str, row_id: int, disk_value: Optional[str]) -> None:
        with connection:
            connection.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        self._delete_disk_files(self._disk_paths(disk_value))

    def _write_disk_files(self, record: Any) -> None:
        files: Optional[Dict[str, bytes]] = getattr(record, "disk_files", None)
        if not self.output_dir or not files:
            return
        for relative_path, data in files.items():
            try:
                path = os.path.join(self.output_dir, relative_path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(data)
            except OSError as error:
                log.warning("[storage] could not write %s: %s", relative_path, error)

    def _delete_disk_files(self, relative_paths: List[str]) -> None:
        if not self.output_dir:
            return
        root = os.path.abspath(self.output_dir)
        for relative_path in relative_paths:
            path = os.path.abspath(os.path.join(root, relative_path))
            try:
                inside_output = os.path.commonpath((root, path)) == root
            except ValueError:
                inside_output = False
            if not inside_output:
                log.warning("[storage] refusing to delete file outside output directory: %s", relative_path)
                continue
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as error:
                log.warning("[storage] could not delete %s: %s", relative_path, error)

    def _run_retention(self, connection: sqlite3.Connection) -> None:
        synced_deleted = 0
        unsynced_deleted = 0
        for table in ("detections", "person_detections"):
            if self.retention_days > 0:
                cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat(sep=" ", timespec="seconds")
                with connection:
                    synced_deleted += connection.execute(
                        f"DELETE FROM {table} WHERE synced = 1 AND detected_at < ?", (cutoff,),
                    ).rowcount
            if self.max_unsynced_days > 0:
                cutoff = (datetime.now() - timedelta(days=self.max_unsynced_days)).isoformat(sep=" ", timespec="seconds")
                with connection:
                    unsynced_deleted += connection.execute(
                        f"DELETE FROM {table} WHERE synced = 0 AND detected_at < ?", (cutoff,),
                    ).rowcount

        files_deleted = 0
        if self.retention_days > 0 and self.output_dir and os.path.isdir(self.output_dir):
            cutoff_ts = time.time() - self.retention_days * 86400
            for folder, _dirs, files in os.walk(self.output_dir):
                for name in files:
                    path = os.path.join(folder, name)
                    try:
                        if os.path.getmtime(path) < cutoff_ts:
                            os.remove(path)
                            files_deleted += 1
                    except OSError as error:
                        log.warning("[storage] could not remove expired file %s: %s", path, error)
        if synced_deleted or unsynced_deleted or files_deleted:
            log.info("[storage] retention cleanup: removed %d synced, %d unsynced row(s), %d file(s)",
                     synced_deleted, unsynced_deleted, files_deleted)

    def status(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "database_path": self.database_path,
            "queue_depth": self._queue.qsize(),
            "writer_alive": bool(self._writer_thread and self._writer_thread.is_alive()),
            "image_writer_alive": bool(self._disk_thread and self._disk_thread.is_alive()),
            "sender_alive": bool(self._sender_thread and self._sender_thread.is_alive()),
        }
        connection = self._connect()
        try:
            for table in ("detections", "person_detections"):
                rows = connection.execute(f"SELECT synced, COUNT(*) FROM {table} GROUP BY synced").fetchall()
                counts = {int(key): int(value) for key, value in rows}
                info[table] = {
                    "total": sum(counts.values()),
                    "waiting": counts.get(0, 0),
                    "delivered": counts.get(1, 0),
                }
        except sqlite3.Error as error:
            info["error"] = str(error)
        finally:
            connection.close()
        return info
