// 자주사용할 함수 정의

// 0으로 채우기
var pad = function (n, width) {
  n = n + '';
  return n.length >= width ? n : new Array(width - n.length + 1).join('0') + n;
}


// 타임스탬프 -> 문자열
export function timeConverter(UNIX_timestamp){
  var a = new Date(UNIX_timestamp);
  //var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var year = a.getFullYear();
  //var month = months[a.getMonth()];
  var date = a.getDate();
  var hour = a.getHours();
  var min = a.getMinutes();
  var sec = a.getSeconds();
  var time = year + '-' + pad((a.getMonth()+1),2) + '-' + pad(date,2) + ' ' +  pad(hour,2) + ':' + pad(min,2) + ':' + pad(sec,2)  ;
  // var time = date + ' ' + month + ' ' + year + ' ' + hour + ':' + min + ':' + sec ;
  return time;
};
// 20181122090000 형식으로
export function timeConverter2(UNIX_timestamp){
  var a = new Date(UNIX_timestamp);
  //var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var year = a.getFullYear();
  //var month = months[a.getMonth()];
  var date = a.getDate();
  var hour = a.getHours();
  var min = a.getMinutes();
  var sec = a.getSeconds();
  var time = year +  pad((a.getMonth()+1),2) +  pad(date,2) +   pad(hour,2) +  pad(min,2) +  pad(sec,2)  ;
  // var time = date + ' ' + month + ' ' + year + ' ' + hour + ':' + min + ':' + sec ;
  return parseInt(time,10);
};

// 랜덤 컬러
export function randomColor(){
  var letters = '0123456789ABCDEF';
  var color = '#';
  for (var i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}
