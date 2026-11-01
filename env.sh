#!/usr/bin/env bash
# Shared runtime and build environment: source /path/to/NeuroSTORM/env.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Run this file with: source ./env.sh\n' >&2
    exit 1
fi

_neurostorm_prepend_path() {
    local variable="$1" directory="$2" current
    current="${!variable-}"
    case ":$current:" in
        *":$directory:"*) return 0 ;;
    esac
    export "$variable=$directory${current:+:$current}"
}

_neurostorm_setup_env() {
    local project_root environment base_prefix conda_activate python_prefix candidate nvcc
    project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
    environment="${NEUROSTORM_ENV:-}"
    if [[ -n "$environment" ]]; then
        environment="$(cd -- "$environment" && pwd)" || return 1
        if [[ ! -x "$environment/bin/python" ]]; then
            printf 'NEUROSTORM_ENV must point to a Python environment.\n' >&2
            return 1
        fi
    else
        for candidate in "$project_root/.env" "$project_root/.venv"; do
            if [[ -f "$candidate/bin/activate" && -x "$candidate/bin/python" ]]; then
                environment="$candidate"
                break
            fi
        done
    fi

    if [[ -n "$environment" && "${VIRTUAL_ENV:-${CONDA_PREFIX:-}}" != "$environment" ]]; then
        base_prefix="$environment"
        if [[ -f "$environment/pyvenv.cfg" ]]; then
            base_prefix="$("$environment/bin/python" -c 'import sys; print(sys.base_prefix)')" || return 1
        fi
        if [[ -d "$base_prefix/conda-meta" ]]; then
            for conda_activate in "$base_prefix/bin/activate" "$base_prefix/../../bin/activate"; do
                if [[ -f "$conda_activate" ]]; then
                    source "$conda_activate" "$base_prefix" || return 1
                    break
                fi
            done
        fi
        if [[ -f "$environment/bin/activate" ]]; then
            source "$environment/bin/activate" || return 1
        elif [[ "${CONDA_PREFIX:-}" != "$environment" ]]; then
            printf 'Activate the selected Conda environment before sourcing env.sh.\n' >&2
            return 1
        fi
    elif [[ -z "$environment" && -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
        printf 'Activate a Conda/virtualenv environment, prepare .env/.venv, or set NEUROSTORM_ENV.\n' >&2
        return 1
    fi

    python_prefix="$(python -c 'import sys; print(sys.prefix)')" || return 1
    export NEUROSTORM_ROOT="$project_root"
    export CONDA_ENV_PATH="${CONDA_PREFIX:-$python_prefix}"
    export PYTHONNOUSERSITE=1
    _neurostorm_prepend_path PYTHONPATH "$project_root"

    # Use project-local defaults, while respecting caller-selected locations.
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$project_root/.cache}"
    export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
    export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
    export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
    export TMPDIR="${TMPDIR:-$project_root/.tmp}"
    # Avoid costly memory compaction when NumPy allocates large scan arrays.
    export NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}"

    # Prefer a prepared compiler, then available environment/system tools.
    if [[ -z "${CC:-}" ]]; then
        for candidate in "$project_root/.compiler/bin/x86_64-conda-linux-gnu-cc" \
            "$python_prefix/bin/x86_64-conda-linux-gnu-cc" gcc clang; do
            if command -v "$candidate" >/dev/null 2>&1; then
                export CC="$candidate"
                break
            fi
        done
    fi
    if [[ -z "${CXX:-}" ]]; then
        for candidate in "$project_root/.compiler/bin/x86_64-conda-linux-gnu-c++" \
            "$python_prefix/bin/x86_64-conda-linux-gnu-c++" g++ clang++; do
            if command -v "$candidate" >/dev/null 2>&1; then
                export CXX="$candidate"
                break
            fi
        done
    fi
    if [[ "${CC:-}" == "$project_root/.compiler/bin/"* || "${CXX:-}" == "$project_root/.compiler/bin/"* ]]; then
        _neurostorm_prepend_path PATH "$project_root/.compiler/bin"
    fi
    if [[ -d "$CONDA_ENV_PATH/x86_64-conda-linux-gnu/lib64" ]]; then
        _neurostorm_prepend_path LD_LIBRARY_PATH "$CONDA_ENV_PATH/x86_64-conda-linux-gnu/lib64"
    fi

    # Respect explicit CUDA settings; otherwise locate an installed toolkit.
    if [[ -z "${CUDA_HOME:-}" ]]; then
        nvcc="$(type -P nvcc || true)"
        if [[ -n "${CUDA_PATH:-}" ]]; then
            export CUDA_HOME="$CUDA_PATH"
        elif [[ -n "$nvcc" ]]; then
            CUDA_HOME="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().parent.parent)' "$nvcc")" || return 1
            export CUDA_HOME
        else
            for candidate in /usr/local/cuda /opt/cuda; do
                if [[ -d "$candidate" ]]; then
                    export CUDA_HOME="$candidate"
                    break
                fi
            done
        fi
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        [[ ! -d "$CUDA_HOME/bin" ]] || _neurostorm_prepend_path PATH "$CUDA_HOME/bin"
        for candidate in "$CUDA_HOME/lib" "$CUDA_HOME/lib64"; do
            if [[ -d "$candidate" ]]; then
                _neurostorm_prepend_path LD_LIBRARY_PATH "$candidate"
                _neurostorm_prepend_path LIBRARY_PATH "$candidate"
            fi
        done
    fi
    # CUDA_VISIBLE_DEVICES and TORCH_CUDA_ARCH_LIST stay under caller control.
    mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
        "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$TMPDIR"
}

_neurostorm_setup_env || {
    unset -f _neurostorm_setup_env _neurostorm_prepend_path
    return 1
}
unset -f _neurostorm_setup_env _neurostorm_prepend_path
