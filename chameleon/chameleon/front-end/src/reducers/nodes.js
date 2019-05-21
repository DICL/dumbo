import * as types from '../actions/Nodes/NodesActionTypes';

// node 리스트들을 redux 에 저장하기 위한 이벤트 처리
const initialState = {
    node_list:[]
};

export default function status(state = initialState, action) {
    switch(action.type) {
        case types.UPDATE_NODE_LIST:
            return {
                ...state,
                node_list: action.node_list,
            };
        default:
            return state;
  }
}
