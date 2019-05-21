import * as types from './ModalsActionTypes';

// modal 창 추가
export function create_modal(modal_data,component_name) {
    return {
        type: types.CREATE_MODAL,
        modal_data,
        component_name,
    };
}

// modal 창 제거
export function remove_modal(index) {
    return {
        type: types.REMOVE_MODAL,
        index
    };
}

// 모든  modal 창 삭제
export function truncate_modal(){
  return {
    type: types.TRUNCATE_MODAL,
  };
}
