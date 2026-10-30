"""provider.mfa entry with a caller-supplied root outside the legacy repository."""
import importlib.util
from pathlib import Path

from host.contracts import ModuleError


def _recover():
    path = Path(__file__).resolve().parent / "recover.py"
    spec = importlib.util.spec_from_file_location("real_align_recover_mfa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Implementation:
    def __init__(self, settings):
        self.settings = settings

    def invoke(self, operation, arguments, api):
        if operation not in ("align_injected", "align_bound"):
            raise ModuleError("UnknownOperation", operation)
        recover = _recover()
        work = Path(arguments["work_dir"])
        recover.assert_outside(work, "provider.mfa workdir")
        if not work.is_dir():
            raise ModuleError("MissingWorkdir", "recovery workdir does not exist",
                              partial={"executed": False})
        try:
            if arguments.get('resources') is not None:
                recover.configure_resources(arguments['resources'])
            if operation == "align_bound":
                recover.PROGRESS = work / "mfa-progress.json"
                report = recover.run_mfa(work, arguments["segment_id"], arguments["source_audio"], include_row=True)
            else:
                report = recover.run_mfa(work)
        except Exception as exc:
            raise ModuleError("MfaFailed", type(exc).__name__ + ": " + str(exc),
                              partial={"executed": False}) from exc
        report["executed"] = True
        report["injected_root"] = True
        return {"report": report}, []
