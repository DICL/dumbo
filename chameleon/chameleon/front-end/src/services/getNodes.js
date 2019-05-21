import axios from 'axios';

// Compute Node 구성을 위한 함수
export function getNodeMatricDataArr() {
	// 호스트 상태
	return axios.get('/api/v1/getNodeMatricDataArr');
}

// Compute Node 구성을 위한 함수
export function getNodeMatricDataArrBak() {
	// 호스트 상태
	return axios.get('/api/v1/getNodeMatricDataArrBak');
}

// Compute Node 구성을 위한 함수
export function getNodeMatricDataArrForHost(host_name) {
	let bodyFormData = new FormData();
	// bodyFormData.set('application_id', application_id);
	bodyFormData.set('host_name', host_name);
	// 호스트 상태
	return axios.post('/api/v1/getNodeMatricDataArrForHost',bodyFormData);
}


// YARNAppMonitorClient 검색하는 함수
export function getYARNAppMonitorClientNodes() {
	// 호스트 상태
	return axios.get('/api/v1/getYARNAppMonitorClientNodes');
}
