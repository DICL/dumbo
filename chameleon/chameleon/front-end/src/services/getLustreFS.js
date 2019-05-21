import axios from 'axios';

// Ambari 에 이ㅛ는 메트릭정보들을 가져오기
export function getLustreMetrics() {
  return axios.get('/api/v1/getLustreMetrics');
}


export function getLustreMetricData(startTime,endTime) {
  return axios.get('/api/v1/getLustreMetricData',{
    params: { startTime: startTime , endTime : endTime}
  });
}

export function new_getLustreMetricData(metricName,startTime,endTime) {
  return axios.get('/api/v1/new_getLustreMetricData',{
    params: { metricName : metricName, startTime: startTime , endTime : endTime}
  });
}


export function getForHostMetricData(metricName,startTime,endTime) {
  return axios.get('/api/v1/getForHostMetricData',{
    params: { metricName : metricName, startTime: startTime , endTime : endTime}
  });
}
