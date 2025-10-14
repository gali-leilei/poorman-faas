set dotenv-load := true
set dotenv-path := "{{justfile_directory()}}/.env"

# default recipe to show all recipes
default:
    @just --list

[group('codegen')]
codegen:
    echo "codegen is disabled"
    # cd {{ justfile_directory() }}/cmd/py_faas && go tool oapi-codegen -config oapi-codegen-config.yaml oapi-spec.yaml

# build the named services
[group('build')]
build binary: codegen
    go build -o {{ justfile_directory() }}/bin/"{{ binary }}" {{ justfile_directory() }}/cmd/"{{ binary }}"
    chmod +x {{ justfile_directory() }}/bin/"{{ binary }}"

# keep the code tidy
[group('build')]
[group('test')]
lint:
    go mod tidy
    golangci-lint run --fix
    golangci-lint run

# run locally
[group('test')]
dev binary: (build binary)
    {{ justfile_directory() }}/bin/"{{ binary }}"

# build containerimage
[group('deploy')]
build-faas:
    podman build -t faas-gateway-app:latest -f {{ justfile_directory() }}/cmd/faas/Dockerfile {{ justfile_directory() }}
    kind load docker-image localhost/faas-gateway-app:latest --name $(kind get clusters)

build-faas-img:
    docker buildx build --platform linux/amd64,linux/arm64 -t  ghcr.io/miromindai/faas-gateway-app:latest  --push \
    -f {{ justfile_directory() }}/cmd/faas/Dockerfile {{ justfile_directory() }}

#	kind load docker-image localhost/faas-gateway-app:latest --name $(kind get clusters)

# deploy service to k8s cluster
[group('deploy')]
deploy-faas: build-faas
    # because tag does not change, we need to manually remove the deployment
    # kubectl delete -f {{ justfile_directory() }}/hack/deployment.yaml
    # rest
    kubectl apply -f {{ justfile_directory() }}/hack/namespace.yaml
    kubectl apply -f {{ justfile_directory() }}/hack/rbac.yaml
    kubectl apply -f {{ justfile_directory() }}/hack/deployment.yaml
    kubectl apply -f {{ justfile_directory() }}/hack/service.yaml

# remove service from k8s cluster
[group('deploy')]
remove-faas:
    kubectl delete -f {{ justfile_directory() }}/hack/service.yaml
    kubectl delete -f {{ justfile_directory() }}/hack/deployment.yaml
    kubectl delete -f {{ justfile_directory() }}/hack/namespace.yaml
