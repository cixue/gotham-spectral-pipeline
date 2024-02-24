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
    while [[ $# > 0 && $1 != "--" ]]; do
        passed_to_gsp+=("$1"); shift
    done

    sdfits=$(${discover_sessions} "$@")
    if [[ $? != 0 ]]; then
        return
    fi

    ${gsp} extract_system_temperature "${passed_to_gsp[@]}" --sdfits ${sdfits[@]}
}

main "$@"
