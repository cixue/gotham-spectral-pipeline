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

    local passed_to_discover_sessions=()
    local argument_mode='normal'
    while [[ $# > 0 ]]; do
        if [[ $1 == --* ]]; then
            if [[ $1 == '--sdfits' ]]; then
                local argument_mode='discover_sessions'
                passed_to_discover_sessions+=("--"); shift
                continue
            elif [[ $1 == '--zenith_opacity' ]]; then
                local argument_mode='zenith_opacity'
                shift
                continue
            else
                local argument_mode='normal'
            fi
        fi
        if [[ ${argument_mode} == 'normal' ]]; then
            echo "Only sdfits and zenith_opacity should be provided."
            exit 1
        elif [[ ${argument_mode} == 'discover_sessions' ]]; then
            passed_to_discover_sessions+=("$1"); shift
        elif [[ ${argument_mode} == 'zenith_opacity' ]]; then
            if [[ -n ${zenith_opacity_directory} ]]; then
                echo "Multiple zenith_opacity specified"
                exit 1
            fi
            local zenith_opacity_directory="$1"; shift
        fi
    done

    if [[ -z ${zenith_opacity_directory} ]]; then
        echo "zenith_opacity must be provided."
        exit 1
    fi

    sdfits=$(${discover_sessions} "${passed_to_discover_sessions[@]}")
    if [[ $? != 0 ]]; then
        return
    fi

    for session in ${sdfits[@]}; do
        echo "Working on ${session}"

        echo "Generating CLEO command..."
        cleo_command=$(${gsp} generate_cleo_command ${session})
        echo "done!"
        echo

        echo "Running CLEO"
        bash -c "mkdir -p ${zenith_opacity_directory}; cd ${zenith_opacity_directory}; ${cleo_command}"
        echo "done!"
        echo
    done
}

main "$@"
