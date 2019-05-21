import { combineReducers } from 'redux';
import status from './status';
import nodes from './nodes';
import modals from './modals';

// redux 사용을 위한 reducers 정의
const reducers = combineReducers({
    status,
    nodes,
    modals,
});

export default reducers;
