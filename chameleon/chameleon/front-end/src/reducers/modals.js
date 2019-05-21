import * as types from '../actions/Modals/ModalsActionTypes';

// node 리스트들을 redux 에 저장하기 위한 이벤트 처리
const initialState = {
    index : 0,
    modal_list:[

    ]
};

export default function modals(state = initialState, action) {
    switch(action.type) {
        // modal 창생성시 데이터 처리
        case types.CREATE_MODAL:
            var tmp = {
              data :action.modal_data,
              component_name : action.component_name ? action.component_name : 'default' ,
              idx : state.index
            };
            return {
                ...state,
                index : state.index + 1,
                modal_list: state.modal_list.concat(tmp)
            };
        // modal 창 삭제시 데이터 처리
        case types.REMOVE_MODAL:
            return {
                ...state,
                modal_list: state.modal_list.filter(modal_item => modal_item.idx !== action.index)
            };

        // modal 창 전체 삭제시 데이터 처리
        case types.TRUNCATE_MODAL:
            return {
                index : 0,
                modal_list:[

                ],
            };
        default:
            return state;
  }
}
