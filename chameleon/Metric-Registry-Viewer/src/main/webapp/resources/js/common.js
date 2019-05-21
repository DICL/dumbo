// 공통함수 정리


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
				  callback();
				} catch (e) {
					console.warn(e);
				}
		},
		function(jqXHR,textStatus,errorThrown) {
			console.error(jqXHR,textStatus,errorThrown);
		}
	);
}