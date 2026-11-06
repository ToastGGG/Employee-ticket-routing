"""允许通过 ``python -m src.employee_ticket_routing`` 执行管理 CLI。"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
