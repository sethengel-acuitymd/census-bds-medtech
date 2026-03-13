# census_bds justfile

@default:
    echo
    echo 'Census BDS Medtech Survival Analysis'
    echo
    echo 'Usage:'
    echo '    just <task>'
    echo
    echo 'Tasks:'
    echo '    sync            sync rye'
    echo '    analyze         run full survival analysis'
    echo '    fmt             format all code'
    echo '    lint            lint and fix'
    echo '    type            type check'
    echo '    verify          format, lint, type-check'
    echo

alias help := default

# list tasks
list:
    just --list --unsorted

# --- workspace ---

# setup development environment
sync:
    rye sync

# run full survival analysis (CLI)
analyze *args:
    rye run analyze {{args}}

# launch interactive dashboard
dashboard:
    streamlit run dashboard.py

# type check
type-check *args:
    rye run pyright {{args}}

alias type := type-check

# format all code
fmt *args:
    rye fmt --all {{args}}

# lint and fix
lint *args:
    rye lint --all --fix {{args}}

# verify: format, lint, type-check
verify *args:
    rye fmt --all {{args}}
    rye lint --all --fix {{args}}
    rye run pyright {{args}}
