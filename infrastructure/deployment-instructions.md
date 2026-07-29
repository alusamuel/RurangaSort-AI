# Cloud Deployment Instructions

Two paths are documented: a **quick path** (any Docker host — EC2, DigitalOcean, Render, Railway) using `docker-compose.yml` as-is, and the **AWS ECS Fargate path** for a more "cloud-native" setup matching `infrastructure/ecs-task-definition.json`. Pick one; both satisfy the assignment's "deploy on a cloud platform" requirement.

## Quick path — any Docker host (e.g. a single EC2 instance)

1. Provision a Linux VM with Docker + Docker Compose installed, and open ports 80 and 8501.
2. `git clone` the repository onto the VM.
3. `cp .env.example .env` and fill in a real `API_KEY` and any AWS/S3 values you plan to use for storage.
4. Train at least one model locally first (`python scripts/prepare_data.py && python -m src.training ...`) or copy a trained `models/active/` directory onto the VM — the containers serve whatever is already in `models/active/` at first boot.
5. `docker compose up --build -d`
6. Nginx listens on port 80 and load-balances across however many `api` replicas you run (`docker compose up -d --scale api=N`); Streamlit is on port 8501, pointed at Nginx via `API_BASE_URL=http://nginx`.
7. Put the VM's public IP/domain into the README's [Production URLs](../README.md#live-urls) section.

This satisfies "deploy on a cloud platform" without needing ECS/EKS — the same `docker-compose.yml` is the artifact, just run on a cloud VM instead of your laptop.

## AWS ECS Fargate path

1. **Build & push images to ECR**

   ```bash
   aws ecr create-repository --repository-name rurangasort-api
   aws ecr create-repository --repository-name rurangasort-ui
   aws ecr create-repository --repository-name rurangasort-worker

   aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com

   docker build -f Dockerfile.api -t rurangasort-api .
   docker tag rurangasort-api:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/rurangasort-api:latest
   docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/rurangasort-api:latest
   # repeat for ui / worker
   ```

2. **Shared model storage** — the API is stateless; a promoted model must be visible to every task. Create an **EFS** file system and mount it into every API/worker task at `/app/models` (see the `volumes` block in `infrastructure/ecs-task-definition.json`) — or push `models/active/` to S3 on every promotion and have each task periodically sync from S3 instead of EFS if you prefer.

3. **Redis** — use **Amazon ElastiCache for Redis** (or a small Redis container in its own ECS service) as the Celery broker/result backend and the shared metrics store; point `REDIS_URL` at its endpoint.

4. **Secrets** — put `API_KEY` (and AWS creds if used) in **AWS Secrets Manager**, referenced from the task definition's `secrets` block (already templated in `ecs-task-definition.json`) — never bake secrets into the image or commit them to the repo.

5. **Register the task definition & service**

   ```bash
   aws ecs register-task-definition --cli-input-json file://infrastructure/ecs-task-definition.json
   aws ecs create-service \
     --cluster rurangasort-cluster \
     --service-name rurangasort-api \
     --task-definition rurangasort-api \
     --desired-count 2 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[YOUR_SUBNETS],securityGroups=[YOUR_SG],assignPublicIp=ENABLED}" \
     --load-balancers "targetGroupArn=YOUR_TARGET_GROUP_ARN,containerName=api,containerPort=8000"
   ```

6. **Application Load Balancer** — create an ALB with a target group pointed at the API service's port 8000; this is the load balancer that fans traffic out across however many API tasks you scale to (the same role Nginx plays in the local Docker Compose setup).

7. **Scaling for the Locust comparison** — change `--desired-count` (1 / 2 / 4) on the ECS service between Locust runs, matching the container-count scenarios in the README's load-test table.

8. **Worker & UI services** — repeat steps 5–6 for the `worker` (no load balancer needed, no inbound port) and `ui` (own ALB or exposed directly) services, pointing the UI's `API_BASE_URL` at the API's ALB DNS name.

9. **Logs & monitoring** — task definitions already ship logs to CloudWatch Logs (`awslogs` driver); optionally add a CloudWatch dashboard/alarm on ALB 5xx rate and target-group latency to complement the app's own `/metrics` endpoint.

10. Fill in the README's [Production URLs](../README.md#live-urls) with the ALB DNS name (or a Route 53 domain pointed at it).

## Common to both paths

- Never commit `.env` or real credentials — `.gitignore` already excludes `.env`.
- Run `python scripts/prepare_data.py` and train/promote at least one model **before** first boot, or `/predict` will correctly 503 until `POST /retrain` produces one.
- After deploying, run the Locust scenarios from the README against the public URL, not `localhost`, to get representative network latency.
