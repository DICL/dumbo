import * as types from '../actions/Status/StatusActionTypes';

// server 상태들을 redux 에 저장하기 위한 이벤트 처리
const initialState = {
    host_starus: {
      "started_count": "0",
      "total_count": "0"
    },
    storage_usage:{
      "CapacityRemaining": "0",
      "CapacityTotal": "0",
      "CapacityUsed": "0"
    },
    yarn_job_list:[

    ],
    yarn_status:{
        "appsSubmitted": 0,
        "appsCompleted": 0,
        "appsPending": 0,
        "appsRunning": 0,
        "appsFailed": 0,
        "appsKilled": 0,
        "reservedMB": 0,
        "availableMB": 0,
        "allocatedMB": 0,
        "reservedVirtualCores": 0,
        "availableVirtualCores": 0,
        "allocatedVirtualCores": 0,
        "containersAllocated": 0,
        "containersReserved": 0,
        "containersPending": 0,
        "totalMB": 0,
        "totalVirtualCores": 0,
        "totalNodes": 2,
        "lostNodes": 0,
        "unhealthyNodes": 0,
        "decommissionedNodes": 0,
        "rebootedNodes": 0,
        "activeNodes": 0
    },

    yarn_job_monitor_list : [],

    alert_list : [],
};

export default function status(state = initialState, action) {
    switch(action.type) {
        case types.UPDATE_HOST_STATUS:
            // console.log(action);
            return {
                ...state,
                host_starus: {
                  ...state.host_starus,
                  "started_count" : action.host_status.started_count,
                  "total_count" : action.host_status.total_count,
                },
            };

        case types.UPDATE_STORAGE_USAGE:
            // console.log(action);
            return {
                ...state,
                storage_usage: {
                  ...state.storage_usage,
                  "CapacityRemaining" : action.storage_usage.CapacityRemaining,
                  "CapacityTotal" : action.storage_usage.CapacityTotal,
                  "CapacityUsed" : action.storage_usage.CapacityUsed,
                },
            };

        case types.UPDATE_YARN_JOB_LIST:
            return {
                ...state,
                yarn_job_list: action.yarn_job_list,
            };

        case types.UPDATE_YARN_STATUS:
            return {
                ...state,
                yarn_status: action.yarn_status,
            };

        case types.UPDATE_YARN_JOB_MONITOR_LIST:
            return {
                ...state,
                yarn_job_monitor_list: action.yarn_job_monitor_list,
            };

        case types.UPDATE_ALERT_LIST:
            return {
                ...state,
                alert_list: action.alert_list,
            };

        default:
            return state;
  }
}
