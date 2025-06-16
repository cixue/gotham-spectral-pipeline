#!/bin/bash

function main() {
    local script_path=$(realpath ${BASH_SOURCE[0]:-${(%):-%x}})
    local script_dirname=$(dirname ${script_path})
    if [[ -e ${script_dirname}/../bin/gsp ]]; then
        local gsp=$(realpath ${script_dirname}/../bin/gsp)
        local discover_sessions=$(realpath ${script_dirname}/../bin/discover_sessions)
    elif [[ -e ${GSPPATH}/bin/gsp ]]; then
        local gsp=${GSPPATH}/bin/gsp
        local discover_sessions=${GSPPATH}/bin/discover_sessions
    else
        echo "Cannot find gsp. Maybe setting GSPPATH?"
        return
    fi

    local passed_to_gsp=()
    local passed_to_discover_sessions=()
    local argument_mode='normal'
    while [[ $# > 0 ]]; do
        if [[ $1 == --* ]]; then
            if [[ $1 == '--sdfits' ]]; then
                local argument_mode='discover_sessions'
                passed_to_discover_sessions+=("--"); shift
                continue
            elif [[ $1 == '--processes' ]]; then
                local argument_mode='processes'
                shift
                continue
            else
                local argument_mode='normal'
            fi
        fi
        if [[ ${argument_mode} == 'normal' ]]; then
            passed_to_gsp+=("$1"); shift
        elif [[ ${argument_mode} == 'discover_sessions' ]]; then
            passed_to_discover_sessions+=("$1"); shift
        elif [[ ${argument_mode} == 'processes' ]]; then
            if [[ -n ${processes} ]]; then
                echo "Multiple numbers of processes specified"
                exit 1
            fi
            local processes="$1"; shift
        fi
    done

    if [[ -z ${processes} ]]; then
        processes=1
    fi

    sdfits=$(${discover_sessions} "${passed_to_discover_sessions[@]}")
    if [[ $? != 0 ]]; then
        return
    fi

    if [[ ${processes} > 1 ]]; then
        for item in ${sdfits[@]}; do
            echo ${item}
        done | \
        xargs --no-run-if-empty --max-args 1 --max-procs ${processes} -I{} \
        bash -c '"$@" --sdfits "$0" > $(basename $0).log 2>&1' \
        {} ${gsp} run_pipeline "${passed_to_gsp[@]}"
    else
        for session in ${sdfits[@]}; do
            ${gsp} run_pipeline "${passed_to_gsp[@]}" --sdfits ${session}
        done
    fi
}

main "$@"
