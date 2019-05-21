import axios from 'axios';


/*
 * chameleon api 호출 을 위한 함수정의
 */

 // ambari api 을 이용하여 호스트 상태를 가져오는 메서드
export function getHostStatus() {
	// 호스트 상태
	return axios.get('/api/v1/getHostStatus');
}

// ambari api 을 통하여 저장소 상태를 가져오는 메서드
export function getStorageUsage(){
	// 저장소 상태
  return axios.get('/api/v1/getStorageUsage');
}

// yarn api 을 통하여 yarn jab list 를 가져오는 메서드
export function getYarnAppList(){
	// yarn joblist
  return axios.get('/api/v1/getYarnAppList');
}

// yarn api 을 통하여 yarn 상태를 가져오는 메서드
export function getYarnStatus(){
	// yarn status
	return axios.get('/api/v1/getYarnStatus');
}

// 실시간으로 yarn 데이터를 가져오는 메서드
export function getYarnJobMonitor(sec){
	let source = axios.CancelToken.source();
	// 3초뒤에 미응답시 취소처리
	setTimeout(() => { source.cancel(); }, sec);
	return axios.get('/api/v1/getYarnJobMonitor',{cancelToken: source.token});
}

// History 내용을 가져오는 메서드
export function getYarnJobHistory(application_id,start_time){
	var bodyFormData = new FormData();
	// bodyFormData.set('application_id', application_id);
	bodyFormData.set('start_time', start_time);
	return axios.post('/api/v1/getYarnJobHistory', bodyFormData);
}

// node 별, 시간별로 데이터 가져오기
export function getYarnJobHistoryPerNode(send_data){
	var bodyFormData = new FormData();
	bodyFormData.set('node', send_data.node);
	bodyFormData.set('start_time', send_data.start_time);
	bodyFormData.set('end_time', send_data.end_time);
	return axios.post('/api/v1/getYarnJobHistoryPerNode', bodyFormData);
}

// ambari metric registry view 의 리스트 가져오기
export function getMetricList(){
	return axios.post('/api/v1/getMetricList');
}


export function getAlert(){
	return axios.get('/api/v1/getAlert');
}
