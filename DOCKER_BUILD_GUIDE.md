# Optimized Docker Build Guide

This guide explains how to use the optimized Docker build process to avoid timeout issues when building the EVIL2ROOT AI trading application.

## Problem Background

Building the Docker image for this application can sometimes take a long time or time out due to:
- The complex installation process for TA-Lib
- Heavy Machine Learning dependencies like TensorFlow and PyTorch
- Large number of Python packages

## Quick Solutions

We've added several optimized build options to help you get past these issues:

### Option 1: Monitored Build with Auto-Fallback

This option monitors the build process and automatically switches to faster build options if timeouts occur:

```bash
./monitor-build.sh --timeout 60  # Timeout in minutes
```

or with make:

```bash
make build-monitored
```

**Best for**: CI/CD environments or when you want the best possible build without manual intervention.

### Option 2: Fast Build with Mock TA-Lib

This option uses a simplified mock implementation of TA-Lib instead of compiling the full library:

```bash
./build-docker.sh --use-mock-talib
```

or with make:

```bash
make build-fast
```

**Best for**: Development and testing when you don't need the full TA-Lib functionality.

### Option 2: Minimal Build (Essentials Only)

This option installs only essential dependencies, skipping the heavyweight ML frameworks:

```bash
./build-docker.sh --essential-only --use-mock-talib
```

or with make:

```bash
make build-minimal
```

**Best for**: Setting up a basic environment quickly when you don't need ML capabilities.

### Option 3: Skip Specific Heavy Dependencies

You can choose to skip specific heavy dependencies like TensorFlow or PyTorch:

```bash
./build-docker.sh --skip-tensorflow --skip-torch
```

**Best for**: When you need most functionality but want to avoid the heaviest dependencies.

### Option 4: Build without Docker Cache

If you're having issues with cached layers, you can build from scratch:

```bash
./build-docker.sh --no-cache
```

or with make:

```bash
make build-no-cache
```

**Best for**: When you suspect there might be issues with cached layers.

## For Production Builds

For full production builds with all features enabled, you should still use:

```bash
docker compose build
```

or

```bash
make build
```

But be prepared for a longer build time. Consider using a CI/CD pipeline with higher timeouts for production builds.

## Troubleshooting Tips

1. **Memory Issues**: If Docker runs out of memory, increase the memory allocation in Docker's settings.

2. **Network Timeouts**: Make sure you have a stable internet connection when building.

3. **Proxy Issues**: If you're behind a corporate firewall, configure Docker to use the proxy.

4. **Docker Logs**: Check the Docker logs for detailed error messages:
   ```bash
   docker build --no-cache --progress=plain -t evil2root_ai:latest .
   ```

5. **Use our troubleshooting script** for automated diagnostics:
   ```bash
   # Check system configuration
   ./docker/troubleshoot-build.sh --check-system
   
   # Try to fix TA-Lib related issues
   ./docker/troubleshoot-build.sh --fix-talib
   
   # Test a minimal build configuration
   ./docker/troubleshoot-build.sh --test-minimal
   ```

6. **Create a pre-built TA-Lib base image** to avoid repeated compilation:
   ```bash
   ./docker/build-talib-base.sh
   ```
   This creates a base image with TA-Lib pre-installed that you can use for all future builds.
