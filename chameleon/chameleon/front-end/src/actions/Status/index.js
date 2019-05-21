import * as types from './StatusActionTypes';

// redux 상태 정의 반환
export function update_host_status(host_status) {
    return {
        type: types.UPDATE_HOST_STATUS,
        host_status
    };
}

export function update_storage_usage(storage_usage) {
    return {
        type: types.UPDATE_STORAGE_USAGE,
        storage_usage
    };
}

export function update_yarn_job_list(yarn_job_list) {
    return {
        type: types.UPDATE_YARN_JOB_LIST,
        yarn_job_list
    };
}

export function update_yarn_status(yarn_status) {
    return {
        type: types.UPDATE_YARN_STATUS,
        yarn_status
    };
}

export function update_yarn_job_monitor_list(yarn_job_monitor_list) {
    return {
        type: types.UPDATE_YARN_JOB_MONITOR_LIST,
        yarn_job_monitor_list
    };
}

export function update_alert_list(alert_list) {
    return {
        type: types.UPDATE_ALERT_LIST,
        alert_list
    };
}
