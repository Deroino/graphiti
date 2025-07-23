# syntax=docker/dockerfile:1.9
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the installer script
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Configure uv for optimal Docker usage - 关键修改：允许使用现有Python
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MCP_SERVER_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd -r app && useradd -r -d /app -g app app

# Copy project files from root context
COPY mcp_server/compat/pyproject.toml ./
COPY mcp_server/compat/uv.lock ./
COPY graphiti_core/ ./graphiti_core/
COPY mcp_server/compat/graphiti_mcp_server.py ./

# Install dependencies using existing Python - 关键修改：构建时安装依赖
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Modify startup command - 使用已安装的环境
CMD ["uv", "run", "--no-sync", "graphiti_mcp_server.py"]
