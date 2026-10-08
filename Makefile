# Local workflow. `make test lint` needs only Python, helm, kubeconform and terraform.
# The cluster targets need Docker + kind and have not been run by the author (see README).
CHART := deploy/helm/llm-platform
TF    := deploy/terraform
CRDS  := https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json
IMAGE := llm-platform-gateway:dev

.PHONY: install test demo lint helm-lint kubeconform tf-validate dashboard image kind-load cluster-up deploy cluster-down port-forward smoke

install:
	pip install -e ".[dev,pgvector]"

test:
	pytest -q

demo:
	python demo/run_demo.py

dashboard:
	python scripts/build_dashboard.py

lint: helm-lint kubeconform tf-validate

helm-lint:
	helm lint $(CHART)
	helm lint $(CHART) -f $(CHART)/values-kind.yaml

kubeconform:
	helm template llm-platform $(CHART) -f $(CHART)/values-kind.yaml \
	  | kubeconform -strict -summary -schema-location default -schema-location '$(CRDS)'
	helm template llm-platform $(CHART) --set vllm.enabled=false,kserve.enabled=true,qdrant.enabled=false,pgvector.enabled=true,vectorStore.kind=pgvector \
	  | kubeconform -strict -summary -schema-location default -schema-location '$(CRDS)'

tf-validate:
	terraform -chdir=$(TF) fmt -check -recursive
	terraform -chdir=$(TF) init -backend=false -input=false
	terraform -chdir=$(TF) validate

image:
	docker build -t $(IMAGE) .

kind-load: image
	kind load docker-image $(IMAGE) --name llm-platform

cluster-up:
	terraform -chdir=$(TF) init
	terraform -chdir=$(TF) apply -target=kind_cluster.this -auto-approve

deploy: cluster-up kind-load
	terraform -chdir=$(TF) apply -auto-approve

port-forward:
	kubectl -n llm port-forward svc/llm-platform-gateway 8080:80

smoke:
	curl -sf localhost:8080/v1/chat/completions -H 'content-type: application/json' \
	  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32}'

cluster-down:
	terraform -chdir=$(TF) destroy -auto-approve
