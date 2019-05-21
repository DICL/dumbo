// 공통함수 정리



Date.prototype.format = function(f) {
    if (!this.valueOf()) return " ";
 
    var weekName = ["일요일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일"];
    var d = this;
     
    return f.replace(/(yyyy|yy|MM|dd|E|hh|mm|ss|a\/p)/gi, function($1) {
        switch ($1) {
            case "yyyy": return d.getFullYear();
            case "yy": return (d.getFullYear() % 1000).zf(2);
            case "MM": return (d.getMonth() + 1).zf(2);
            case "dd": return d.getDate().zf(2);
            case "E": return weekName[d.getDay()];
            case "HH": return d.getHours().zf(2);
            case "hh": return ((h = d.getHours() % 12) ? h : 12).zf(2);
            case "mm": return d.getMinutes().zf(2);
            case "ss": return d.getSeconds().zf(2);
            case "a/p": return d.getHours() < 12 ? "오전" : "오후";
            default: return $1;
        }
    });
};
 
String.prototype.string = function(len){var s = '', i = 0; while (i++ < len) { s += this; } return s;};
String.prototype.zf = function(len){return "0".string(len - this.length) + this;};
Number.prototype.zf = function(len){return this.toString().zf(len);};

Number.prototype.comma = function(){
    if(this==0) return 0;
 
    var reg = /(^[+-]?\d+)(\d{3})/;
    var n = (this + '');
 
    while (reg.test(n)) n = n.replace(reg, '$1' + ',' + '$2');
 
    return n;
};


function getDates(startDate, endDate, dateFormat) {
  var dateArray = new Array();
  var currentDate = startDate;
  while (currentDate <= endDate) {
    dateArray.push(currentDate.format(dateFormat));
    currentDate = currentDate.addDays(1);
  }
  return dateArray;
}


/**
 * 로딩화면이 나오는 메서드
 * @param loading_text 로딩문구
 * @returns
 */
function showLoading(loading_text) {
	if(!window["start_time"]){
		window["start_time"] = new Date().getTime();
	}
	var taget = ".pure-loarding";
	$(taget+" .message").text("");
	$(taget+" .message").text(loading_text);
	$(taget).show();
}

/**
 * 로딩화면을 숨기고 로딩시간을 출력하는 메서드
 * @param loading_text
 * @returns
 */
function hideLoading() {
	window["end_time"] = new Date().getTime();
	var taget = ".pure-loarding";
	$(taget).hide();
	var total_time = (window["end_time"] - window["start_time"]) / 1000;
	console.log("Loading Time (sec) ==>",total_time);
	window["start_time"] = undefined;
	window["end_time"] = undefined;
}

/**
 * 로딩문구를 변결하는 메서드
 * @param loading_text
 * @returns
 */
function changeLoadingText(loading_text) {
	var taget = ".pure-loarding";
	$(taget+" .message").text("");
	$(taget+" .message").text(loading_text);
}

/**
 * 템플릿 페이지를 읽어와서 화면에 출력하는 메서드
 * @param target_object 화면에 그릴 DOM 겍체
 * @param templete_name 불려올 템플릿 파일명
 * @param data 참고할 데이터
 * @param callback 콜백함수
 * @returns
 */
function templete_html (target_object,templete_name,data,callback) {
	changeLoadingText("grid templete page");
	$.ajax({
		url: contextPath +"/template/"+ templete_name + ".hbs",
		type: "GET",
		data: {}
	}).then(
		function ( template_page, textStatus, jqXHR) {
			var template=Handlebars.compile(template_page);
			target_object.innerHTML=template(data);
			 try {
				  callback(data);
				} catch (e) {
					console.warn(e);
				}
		},
		function(jqXHR,textStatus,errorThrown) {
			console.error(jqXHR,textStatus,errorThrown);
		}
	);
}


jQuery.loadScript = function (js_name, callback) {
    jQuery.ajax({
        url: contextPath +"/js/"+js_name+'.js',
        dataType: 'script',
        success: callback,
        async: true
    });
}