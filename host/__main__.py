"""Minimal startup: one registered plan entry, no test/workflow ordering here."""
import argparse
import json
import pathlib
import sys

from .contracts import ContractError, ModuleError
from .runtime import Host, HOST_VERSION


def main():
    parser = argparse.ArgumentParser(description="Temporary controlled B00 host")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = pathlib.Path(__file__).resolve().parent.parent
    bootstrap = sys.modules["host"]
    # 0.1.5: dispatch only plans registered in module-registry/plans.json.
    registered = [p for p in bootstrap._plan_registry["plans"] if p["plan"] == args.plan]
    if len(registered) != 1:
        parser.error("plan is not registered in " + bootstrap.PLAN_REGISTRY)
    registration = registered[0]
    host = None
    try:
        plan = json.loads(bootstrap.guarded_path(project, args.plan, ContractError).read_text("utf-8"))
        if plan["entry"] != registration["entry"]:
            raise ContractError("plan entry must match the registered bootstrap root grant")
        host = Host(project, args.output, registration)
        task = {"run_id": "bootstrap-run", "binding_id": "bootstrap"}
        try:
            result = host.invoke(host.envelope(task, plan["entry"], {"plan_id": plan["id"]}))
        finally:
            cleanup_error = None
            try:
                host.end_runs()
            except ModuleError as exc:
                cleanup_error = exc.error
                raise
            finally:
                if host.resources.records or "resource_pack" in host.environment:
                    path = bootstrap.guarded_path(host.output, "resource-handles.json", ContractError)
                    with path.open("x", encoding="utf-8", newline="\n") as output:
                        evidence = {"host_version": HOST_VERSION, "resource_handles": host.resources.snapshot(),
                                    "resource_pack": host.environment.get("resource_pack", [])}
                        if cleanup_error is not None:
                            evidence["cleanup_error"] = cleanup_error
                        json.dump(host.redact(evidence), output, ensure_ascii=False, indent=2)
                        output.write("\n")
        exit_code = 0 if result["outcome"] == "returned" else 1
    except (ContractError, ModuleError, OSError, KeyError, ValueError) as exc:
        error = {"startup_error": type(exc).__name__, "message": str(exc)}
        print(json.dumps(host.redact(error) if host is not None else error), file=sys.stderr)
        return 2
    # A completed run's conclusion is independent of transport/console failures.
    try:
        print(json.dumps(host.redact(result), ensure_ascii=True))
        sys.stdout.flush()
    except (OSError, UnicodeError) as exc:
        sys.stdout = None  # Do not let shutdown retry a failed buffered flush.
        try:
            print(json.dumps(host.redact({"result_output_error": type(exc).__name__,
                              "run_exit_code": exit_code, "message": str(exc)}),
                             ensure_ascii=True), file=sys.stderr)
        except (OSError, UnicodeError):
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
