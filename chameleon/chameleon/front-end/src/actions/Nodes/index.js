import * as types from './NodesActionTypes';

// redux 상태 정의 반환
export function update_node_list(node_list) {
    return {
        type: types.UPDATE_NODE_LIST,
        node_list
    };
}
