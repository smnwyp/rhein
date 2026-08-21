"""Index + gzip-artifact persistence for resumable research reports."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from alpha_agent.errors import AlphaAgentError

from .models import SavedResearchReport, SavedResearchReportSummary, VisualObservation


class ResearchHistoryError(AlphaAgentError):
    code = "research_history_error"


class JsonResearchReportHistory:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._index_path = path.with_name(f"{path.stem}_index.json")
        self._artifact_directory = path.parent / "saved_research_report_artifacts"

    def list_for_run(self, run_id: UUID) -> list[SavedResearchReportSummary]:
        return sorted((item for item in self._read_index() if item.run_id == run_id), key=lambda item: item.updated_at, reverse=True)

    def find_by_input(self, input_fingerprint: str) -> SavedResearchReport | None:
        summary = next((item for item in self._read_index() if item.input_fingerprint == input_fingerprint), None)
        return self.load(summary.report_id) if summary else None

    def observations_by_checksum(self, checksums: set[str]) -> dict[str, VisualObservation]:
        """Reuse schema-validated, focus-independent image observations.

        Chart PNGs are not persisted.  Their checksum is retained in each
        report, and this lookup only reads existing report artifacts.
        """
        found: dict[str, VisualObservation] = {}
        for summary in self._read_index():
            if not checksums - set(found):
                break
            report = self.load(summary.report_id)
            for observation in report.observations:
                if observation.chart_checksum in checksums:
                    found.setdefault(observation.chart_checksum, observation)
        return found

    def load(self, report_id: UUID) -> SavedResearchReport:
        summary = next((item for item in self._read_index() if item.report_id == report_id), None)
        if summary is None:
            raise ResearchHistoryError("saved research report was not found", details={"report_id": str(report_id)})
        try:
            with gzip.open(self._artifact_path(summary), "rt", encoding="utf-8") as artifact:
                return SavedResearchReport.model_validate_json(artifact.read())
        except (OSError, ValueError) as error:
            raise ResearchHistoryError("saved research report artifact cannot be read", details={"report_id": str(report_id)}) from error

    def save(self, report: SavedResearchReport) -> SavedResearchReport:
        summary = SavedResearchReportSummary.from_report(report)
        self._write_artifact(report, summary)
        entries = [item for item in self._read_index() if item.report_id != report.report_id]
        entries.append(summary)
        self._atomic_write(self._index_path, json.dumps([item.model_dump(mode="json") for item in entries], ensure_ascii=False, indent=2).encode())
        return report

    def delete(self, report_id: UUID) -> None:
        entries = self._read_index()
        removed = [item for item in entries if item.report_id == report_id]
        if not removed:
            raise ResearchHistoryError("saved research report was not found", details={"report_id": str(report_id)})
        self._artifact_path(removed[0]).unlink(missing_ok=True)
        self._atomic_write(self._index_path, json.dumps([item.model_dump(mode="json") for item in entries if item.report_id != report_id], ensure_ascii=False, indent=2).encode())

    def delete_for_strategy(self, strategy_id: UUID) -> None:
        for summary in [item for item in self._read_index() if item.strategy_id == strategy_id]:
            self.delete(summary.report_id)

    def _read_index(self) -> list[SavedResearchReportSummary]:
        if not self._index_path.exists():
            return []
        try:
            return [SavedResearchReportSummary.model_validate_json(json.dumps(item)) for item in json.loads(self._index_path.read_text(encoding="utf-8"))]
        except (OSError, ValueError) as error:
            raise ResearchHistoryError("saved research report index cannot be read", details={"path": str(self._index_path)}) from error

    def _write_artifact(self, report: SavedResearchReport, summary: SavedResearchReportSummary) -> None:
        self._artifact_directory.mkdir(parents=True, exist_ok=True)
        payload = report.model_dump_json().encode("utf-8")
        path = self._artifact_path(summary)
        try:
            with NamedTemporaryFile("wb", dir=path.parent, delete=False) as temporary:
                with gzip.GzipFile(fileobj=temporary, mode="wb", compresslevel=9, mtime=0) as compressed:
                    compressed.write(payload)
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        except OSError as error:
            raise ResearchHistoryError("saved research report cannot be written", details={"path": str(path)}) from error

    def _artifact_path(self, summary: SavedResearchReportSummary) -> Path:
        name = Path(summary.artifact_file).name
        if name != summary.artifact_file or not name.endswith(".json.gz"):
            raise ResearchHistoryError("saved research report artifact path is invalid")
        return self._artifact_directory / name

    @staticmethod
    def _atomic_write(path: Path, contents: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", dir=path.parent, delete=False) as temporary:
            temporary.write(contents)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
