# Kubernetes Health Checks

This guide explains how the platform team configures health checks for services running on Kubernetes.

## Liveness Probes

A liveness probe tells Kubernetes when a container is stuck and must be restarted. Our services expose GET /healthz for this purpose. The probe runs every 10 seconds, and the container is restarted after 3 consecutive failures. Never check downstream dependencies such as the database in a liveness probe, because a database outage would then restart every pod at once.

## Readiness Probes

A readiness probe decides whether a pod receives traffic from its Service. Our services expose GET /readyz, which does check the database and cache connections. A pod that fails readiness is removed from load balancing but is not restarted. Readiness is checked every 5 seconds.

## Startup Probes

Services with slow initialization, such as the search indexer that loads a 2 GB model on boot, use a startup probe. Liveness and readiness checks are disabled until the startup probe succeeds. The startup probe allows up to 30 failures at 10 second intervals, giving the container five minutes to start.

## Resource Limits

Every container must declare CPU and memory requests. The default memory limit is 512 MiB. A container that exceeds its memory limit is killed with reason OOMKilled, which is different from a failed liveness probe.
