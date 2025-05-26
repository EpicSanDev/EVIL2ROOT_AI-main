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

## ARM64-Specific Builds (Apple Silicon)

If you're using an ARM64-based machine (like M1/M2/M3 Macs), we've created specialized build processes:

### Option 1: Optimized ARM64 Build

```bash
./docker/build-arm64.sh
```

or

```bash
make build-arm64
```

**Best for**: Standard build optimized for ARM64 architecture.

### Option 2: Fast ARM64 Build with Mock TA-Lib

```bash
./docker/build-arm64.sh --use-mock-talib
```

or

```bash
make build-arm64-mock
```

**Best for**: Development on Apple Silicon when you need a fast build and don't need full TA-Lib functionality.

### Option 3: Minimal ARM64 Build

```bash
./docker/build-arm64.sh --essential-only
```

or

```bash
make build-arm64-minimal
```

**Best for**: Quick setup with only essential features on ARM64.

### ARM64 Build Features

Our ARM64-specific build process includes:

1. **Resource Pre-check**: Checks your system memory and disk space and provides recommendations
2. **Architecture-specific Optimization**: Uses optimized build flags for ARM64
3. **Automatic Fallbacks**: If a build fails or times out, it automatically tries less resource-intensive options
4. **Timeout Management**: Built-in timeouts to prevent hanging builds
5. **Mock TA-Lib Implementation**: Uses a specially optimized mock implementation for ARM64

### Apple Silicon Specific Notes

For M1/M2/M3 Mac users:
- Make sure Docker Desktop is set to use the "Use Rosetta for x86/amd64 emulation" option if you need to build x86 images
- For maximum performance, use the native ARM64 build process
- Our mock TA-Lib implementation includes optimized versions of key indicators (SMA, EMA, RSI, MACD, BBANDS) that perform well on Apple Silicon

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
