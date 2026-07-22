from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import aiosqlite
import sqlglot
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)

from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    SQLiteSession,
    function_tool,
)


# ============================================================
# Shared models
# ============================================================


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetArtifact(StrictModel):
    dataset_id: str
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]


class DataQueryResult(StrictModel):
    dataset_id: str
    sql: str
    row_count: int
    columns: list[str]
    machine_ids: list[str] = Field(default_factory=list)
    summary: str


class DashboardResult(StrictModel):
    dashboard_id: str
    dashboard_url: str | None = None
    summary: str


class RemoteActionResult(StrictModel):
    action_id: str
    status: str
    succeeded: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


class SQLValidationResult(StrictModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class QueryObservation(StrictModel):
    success: bool
    sql: str
    row_count: int = 0
    columns: list[str] = Field(default_factory=list)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    error: str | None = None


class SQLAgentOutput(StrictModel):
    success: bool
    final_sql: str | None = None
    summary: str
    failure_reason: str | None = None


# ============================================================
# Artifact store
#
# Conversation memory should contain dataset IDs, not entire
# datasets. Actual query results are stored separately.
# ============================================================


class ArtifactStore:
    def __init__(self, db_path: str = "artifacts.db") -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    rows_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            await db.commit()

    async def save_dataset(
        self,
        rows: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> str:
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO datasets (
                    dataset_id,
                    rows_json,
                    metadata_json
                )
                VALUES (?, ?, ?)
                """,
                (
                    dataset_id,
                    json.dumps(rows, default=str),
                    json.dumps(metadata, default=str),
                ),
            )
            await db.commit()

        return dataset_id

    async def get_dataset(
        self,
        dataset_id: str,
    ) -> DatasetArtifact:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT rows_json, metadata_json
                FROM datasets
                WHERE dataset_id = ?
                """,
                (dataset_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            raise ValueError(
                f"Unknown dataset_id: {dataset_id}"
            )

        return DatasetArtifact(
            dataset_id=dataset_id,
            rows=json.loads(row[0]),
            metadata=json.loads(row[1]),
        )


# ============================================================
# MCP gateway
#
# Capabilities call MCP tools. MCP tools are not exposed
# directly to the master agent in this design.
# ============================================================


class MCPGateway:
    def __init__(
        self,
        server_urls: dict[str, str],
    ) -> None:
        self.server_urls = server_urls

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            url = self.server_urls[server_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown MCP server: {server_name}"
            ) from exc

        async with streamable_http_client(url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:
                await session.initialize()

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                )

        if getattr(result, "isError", False):
            error_text = self._extract_text(result)
            raise RuntimeError(
                error_text or f"MCP tool failed: {tool_name}"
            )

        structured = getattr(
            result,
            "structuredContent",
            None,
        )

        if structured is not None:
            return dict(structured)

        return {
            "text": self._extract_text(result),
        }

    @staticmethod
    def _extract_text(result: Any) -> str:
        parts: list[str] = []

        for item in getattr(result, "content", []):
            if isinstance(item, types.TextContent):
                parts.append(item.text)

        return "\n".join(parts)


# ============================================================
# Application context for the master agent
#
# This is dependency injection. It is not automatically shown
# to the LLM.
# ============================================================


@dataclass
class AppContext:
    user_id: str
    database: AsyncEngine
    artifacts: ArtifactStore
    mcp: MCPGateway
    allowed_tables: set[str]


# ============================================================
# SQL agent local context
#
# This state exists only during one query_data invocation.
# ============================================================


@dataclass
class SQLContext:
    database: AsyncEngine
    allowed_tables: set[str]

    max_rows: int = 200
    max_execution_attempts: int = 3

    execution_attempts: int = 0
    last_sql: str | None = None
    last_rows: list[dict[str, Any]] = field(
        default_factory=list
    )
    last_columns: list[str] = field(
        default_factory=list
    )


# ============================================================
# SQL validation
# ============================================================


def validate_sql_text(
    sql: str,
    allowed_tables: set[str],
) -> SQLValidationResult:
    errors: list[str] = []

    try:
        statements = sqlglot.parse(
            sql,
            read="sqlite",
        )
    except Exception as exc:
        return SQLValidationResult(
            valid=False,
            errors=[f"SQL parse error: {exc}"],
        )

    if len(statements) != 1:
        errors.append(
            "Exactly one SQL statement is allowed."
        )
        return SQLValidationResult(
            valid=False,
            errors=errors,
        )

    statement = statements[0]

    forbidden_type_names = (
        "Insert",
        "Update",
        "Delete",
        "Create",
        "Drop",
        "Alter",
        "Command",
        "Merge",
        "TruncateTable",
    )

    forbidden_types = tuple(
        getattr(sqlglot.exp, name)
        for name in forbidden_type_names
        if hasattr(sqlglot.exp, name)
    )

    for forbidden_type in forbidden_types:
        if statement.find(forbidden_type) is not None:
            errors.append(
                f"{forbidden_type.__name__} is not allowed."
            )

    has_select = any(
        True
        for _ in statement.find_all(
            sqlglot.exp.Select
        )
    )

    if not has_select:
        errors.append(
            "The query must contain a SELECT statement."
        )

    referenced_tables = {
        table.name
        for table in statement.find_all(
            sqlglot.exp.Table
        )
    }

    unauthorized_tables = (
        referenced_tables - allowed_tables
    )

    if unauthorized_tables:
        errors.append(
            "Unauthorized or unknown tables: "
            + ", ".join(
                sorted(unauthorized_tables)
            )
        )

    return SQLValidationResult(
        valid=not errors,
        errors=errors,
    )


# ============================================================
# SQL-agent tools
# ============================================================


@function_tool
async def get_database_schema(
    ctx: RunContextWrapper[SQLContext],
) -> str:
    """
    Return the authorized database schema.

    Always call this before generating SQL.
    """

    def inspect_schema(sync_connection: Any) -> str:
        inspector = inspect(sync_connection)
        output: list[str] = []

        for table_name in sorted(
            ctx.context.allowed_tables
        ):
            if table_name not in inspector.get_table_names():
                continue

            output.append(f"TABLE {table_name}")

            for column in inspector.get_columns(
                table_name
            ):
                output.append(
                    f"  {column['name']}: "
                    f"{column['type']}"
                )

        return "\n".join(output)

    async with ctx.context.database.connect() as conn:
        return await conn.run_sync(inspect_schema)


@function_tool
async def validate_sql(
    ctx: RunContextWrapper[SQLContext],
    sql: str,
) -> SQLValidationResult:
    """
    Validate a candidate read-only SQL statement.

    Do not execute SQL until this tool returns valid=true.
    """

    return validate_sql_text(
        sql,
        ctx.context.allowed_tables,
    )


@function_tool
async def execute_readonly_sql(
    ctx: RunContextWrapper[SQLContext],
    sql: str,
) -> QueryObservation:
    """
    Execute a validated, read-only SQL statement.

    The complete result is stored locally. A sample is returned
    so that the SQL agent can judge whether the result answers
    the user's question.
    """

    validation = validate_sql_text(
        sql,
        ctx.context.allowed_tables,
    )

    if not validation.valid:
        return QueryObservation(
            success=False,
            sql=sql,
            error=(
                "Validation failed: "
                + "; ".join(validation.errors)
            ),
        )

    ctx.context.execution_attempts += 1

    if (
        ctx.context.execution_attempts
        > ctx.context.max_execution_attempts
    ):
        return QueryObservation(
            success=False,
            sql=sql,
            error=(
                "Maximum SQL execution attempts exceeded."
            ),
        )

    clean_sql = sql.strip().rstrip(";")

    # Defense in depth: enforce a hard output cap even when
    # the generated query omitted LIMIT.
    limited_sql = f"""
        SELECT *
        FROM ({clean_sql}) AS agent_query
        LIMIT {ctx.context.max_rows + 1}
    """

    try:
        async with asyncio.timeout(10):
            async with ctx.context.database.connect() as conn:
                result = await conn.execute(
                    text(limited_sql)
                )

                columns = list(result.keys())
                rows = [
                    dict(row._mapping)
                    for row in result.fetchall()
                ]

    except Exception as exc:
        return QueryObservation(
            success=False,
            sql=sql,
            error=f"Database execution error: {exc}",
        )

    truncated = (
        len(rows) > ctx.context.max_rows
    )
    rows = rows[: ctx.context.max_rows]

    ctx.context.last_sql = sql
    ctx.context.last_rows = rows
    ctx.context.last_columns = columns

    return QueryObservation(
        success=True,
        sql=sql,
        row_count=len(rows),
        columns=columns,
        sample_rows=rows[:10],
        truncated=truncated,
    )


# ============================================================
# Bounded SQL agent
# ============================================================


sql_agent = Agent[SQLContext](
    name="Bounded SQL query agent",
    instructions="""
You convert a user's data request into a safe, read-only SQL query.

Required process:

1. Call get_database_schema before producing SQL.
2. Generate exactly one read-only SELECT query.
3. Call validate_sql.
4. If validation fails, use the returned errors to repair the
   SQL and validate it again.
5. Only after validation succeeds, call execute_readonly_sql.
6. Inspect the execution outcome and sample rows.
7. If execution fails, repair the SQL, validate it again, and
   retry.
8. If execution succeeds but the result does not answer the
   user's request, revise, validate, and retry.
9. Finish when the result adequately answers the request or
   when further progress is impossible.

Never invent tables or columns.
Never perform INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
administrative commands, or multiple statements.
Do not claim success unless execute_readonly_sql succeeded.
""",
    tools=[
        get_database_schema,
        validate_sql,
        execute_readonly_sql,
    ],
    output_type=SQLAgentOutput,
)


# ============================================================
# High-level capability 1: query data
#
# The master agent sees only query_data, not get_schema,
# validate_sql or execute_sql.
# ============================================================


@function_tool
async def query_data(
    ctx: RunContextWrapper[AppContext],
    request: str = Field(
        description=(
            "A precise natural-language description of the "
            "data that should be retrieved."
        )
    ),
) -> DataQueryResult:
    """
    Query authorized operational data.

    Use this for database searches, aggregations, filtering,
    comparisons and other questions requiring database data.
    """

    sql_context = SQLContext(
        database=ctx.context.database,
        allowed_tables=ctx.context.allowed_tables,
    )

    nested_result = await Runner.run(
        starting_agent=sql_agent,
        input=request,
        context=sql_context,
        max_turns=14,
    )

    decision = nested_result.final_output

    if not isinstance(decision, SQLAgentOutput):
        decision = SQLAgentOutput.model_validate(
            decision
        )

    if (
        not decision.success
        or sql_context.last_sql is None
    ):
        raise RuntimeError(
            decision.failure_reason
            or "The SQL agent could not complete the query."
        )

    dataset_id = (
        await ctx.context.artifacts.save_dataset(
            rows=sql_context.last_rows,
            metadata={
                "request": request,
                "sql": sql_context.last_sql,
                "columns": sql_context.last_columns,
            },
        )
    )

    machine_ids = [
        str(row["machine_id"])
        for row in sql_context.last_rows
        if row.get("machine_id") is not None
    ]

    return DataQueryResult(
        dataset_id=dataset_id,
        sql=sql_context.last_sql,
        row_count=len(sql_context.last_rows),
        columns=sql_context.last_columns,
        machine_ids=machine_ids,
        summary=decision.summary,
    )


# ============================================================
# High-level capability 2: dashboard
# ============================================================


@function_tool
async def create_dashboard(
    ctx: RunContextWrapper[AppContext],
    dataset_id: str = Field(
        description=(
            "The dataset ID returned by query_data or from "
            "an earlier conversation turn."
        )
    ),
    instruction: str = Field(
        description=(
            "What the dashboard should communicate and which "
            "metrics or dimensions it should emphasize."
        )
    ),
) -> DashboardResult:
    """
    Create a dashboard from an existing dataset.

    Do not use this tool without a valid dataset_id.
    """

    dataset = await ctx.context.artifacts.get_dataset(
        dataset_id
    )

    raw_result = await ctx.context.mcp.call_tool(
        server_name="dashboard",
        tool_name="create_dashboard",
        arguments={
            "dataset_id": dataset.dataset_id,
            "rows": dataset.rows,
            "metadata": dataset.metadata,
            "instruction": instruction,
        },
    )

    return DashboardResult.model_validate(
        raw_result
    )


# ============================================================
# High-level capability 3: remote action
#
# needs_approval=True pauses the outer master-agent run before
# this function is actually executed.
# ============================================================


@function_tool(needs_approval=True)
async def execute_remote_action(
    ctx: RunContextWrapper[AppContext],
    action: Literal[
        "restart",
        "stop",
        "start",
        "clear_cache",
    ],
    machine_ids: list[str] = Field(
        min_length=1,
        description=(
            "Explicit machine IDs. Never use descriptions such "
            "as 'all machines' or invent IDs."
        ),
    ),
) -> RemoteActionResult:
    """
    Execute an action on one or more remote machines.

    Call this only when the user explicitly requested the
    action. Execution requires human approval.
    """

    raw_result = await ctx.context.mcp.call_tool(
        server_name="machines",
        tool_name="execute_machine_action",
        arguments={
            "requested_by": ctx.context.user_id,
            "action": action,
            "machine_ids": machine_ids,
            "idempotency_key": (
                f"{ctx.context.user_id}:"
                f"{action}:"
                f"{','.join(sorted(machine_ids))}"
            ),
        },
    )

    return RemoteActionResult.model_validate(
        raw_result
    )


# ============================================================
# Capability registry
#
# Agents SDK FunctionTool objects already contain:
# - name
# - description
# - JSON input schema
# - implementation
#
# Therefore the registry can remain thin.
# ============================================================


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, Any] = {}

    def register(self, *capabilities: Any) -> None:
        for capability in capabilities:
            if capability.name in self._capabilities:
                raise ValueError(
                    "Duplicate capability: "
                    f"{capability.name}"
                )

            self._capabilities[
                capability.name
            ] = capability

    def tools(self) -> list[Any]:
        return list(
            self._capabilities.values()
        )


registry = CapabilityRegistry()

registry.register(
    query_data,
    create_dashboard,
    execute_remote_action,
)


# ============================================================
# Master agent
# ============================================================


master_agent = Agent[AppContext](
    name="Operations master agent",
    instructions="""
You own the user conversation and decide which high-level
capability to call next based on the current request and all
previous capability results.

Available behavior:

- Use query_data for requests requiring database information.
- Use create_dashboard only when you have a valid dataset_id.
- Use execute_remote_action only when the user explicitly asks
  for an operational action.
- Never invent dataset IDs, machine IDs or capability results.
- When one request contains several goals, work incrementally.

Examples:

"Find offline machines and create a dashboard"
1. Call query_data.
2. Observe its dataset_id.
3. Call create_dashboard using that dataset_id.
4. Return a combined answer.

"Find offline machines and restart them"
1. Call query_data.
2. Observe its machine_ids.
3. If no machines were returned, do not call the action.
4. Otherwise call execute_remote_action with those exact IDs.
5. The runtime will request human approval.
6. Report the execution result.

"Create a dashboard from the previous results"
Use the dataset_id already present in conversation history.
Do not repeat the database query unless the existing dataset
cannot satisfy the request.

After each capability result, reassess what remains to be done.
Stop when the user's goal is complete, clarification is needed,
or no safe progress is possible.
""",
    tools=registry.tools(),
)


# ============================================================
# Demo database
# ============================================================


async def initialize_demo_database(
    engine: AsyncEngine,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS machines (
                    machine_id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    status TEXT NOT NULL,
                    offline_minutes INTEGER NOT NULL,
                    cpu_usage REAL NOT NULL
                )
                """
            )
        )

        count_result = await conn.execute(
            text(
                "SELECT COUNT(*) FROM machines"
            )
        )

        count = count_result.scalar_one()

        if count == 0:
            await conn.execute(
                text(
                    """
                    INSERT INTO machines (
                        machine_id,
                        hostname,
                        status,
                        offline_minutes,
                        cpu_usage
                    )
                    VALUES
                        ('machine-001', 'web-01',
                         'online', 0, 43.2),
                        ('machine-002', 'web-02',
                         'offline', 95, 0),
                        ('machine-003', 'worker-01',
                         'offline', 130, 0),
                        ('machine-004', 'db-01',
                         'online', 0, 88.1)
                    """
                )
            )


# ============================================================
# Chat loop with HITL approval
# ============================================================


async def run_chat() -> None:
    database = create_async_engine(
        "sqlite+aiosqlite:///operations.db"
    )

    await initialize_demo_database(database)

    artifacts = ArtifactStore(
        db_path="artifacts.db"
    )
    await artifacts.initialize()

    mcp = MCPGateway(
        {
            "dashboard": os.environ.get(
                "DASHBOARD_MCP_URL",
                "http://localhost:8100/mcp",
            ),
            "machines": os.environ.get(
                "MACHINE_MCP_URL",
                "http://localhost:8200/mcp",
            ),
        }
    )

    context = AppContext(
        user_id="user-123",
        database=database,
        artifacts=artifacts,
        mcp=mcp,
        allowed_tables={"machines"},
    )

    # Maintains the master agent's conversation history across
    # user turns.
    session = SQLiteSession(
        "user-123:operations-chat",
        "conversation_history.db",
    )

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {
                "quit",
                "exit",
            }:
                break

            result = await Runner.run(
                starting_agent=master_agent,
                input=user_input,
                context=context,
                session=session,
                max_turns=16,
            )

            # A remote action tool pauses here before execution.
            while result.interruptions:
                state = result.to_state()

                for interruption in result.interruptions:
                    tool_name = getattr(
                        interruption,
                        "tool_name",
                        "unknown_tool",
                    )
                    arguments = getattr(
                        interruption,
                        "arguments",
                        {},
                    )

                    print(
                        "\nApproval required:"
                        f"\nTool: {tool_name}"
                        f"\nArguments: {arguments}"
                    )

                    answer = input(
                        "Approve? [y/N]: "
                    ).strip().lower()

                    if answer == "y":
                        state.approve(interruption)
                    else:
                        state.reject(
                            interruption,
                            rejection_message=(
                                "The user rejected the "
                                "remote machine action."
                            ),
                        )

                # Resume the original master-agent loop from
                # exactly where it paused.
                result = await Runner.run(
                    starting_agent=master_agent,
                    input=state,
                    session=session,
                    max_turns=16,
                )

            print(f"\nAssistant: {result.final_output}")

    finally:
        await database.dispose()


if __name__ == "__main__":
    asyncio.run(run_chat())