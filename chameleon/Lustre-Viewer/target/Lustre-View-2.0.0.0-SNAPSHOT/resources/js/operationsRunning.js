$(function(){
	operationsRunning.listener();
});

var operationsRunning = {
		// interval id
		setInterval_jobLog : null,
		// 최근에 불려온 table num
		max_num:0,
		// log id
		row_key:null,
		// 필터링할 log type
		filter_log : {
			'command' : true,
			'info' : true,
			'error' : true,
		},
		
		
		/**
		 * 서버에서 로그 리스트 불려오기
		 */
		viewLogList : function(callback) {
			$.ajax({
				url: contextPath+'/api/v1/lustre/getLogList',
				type: "POST",
				data: {}
			}).then(
				function ( data, textStatus, jqXHR) {
					data = _.map(data,function(log_info){
						log_info['create_date'] = new Date(log_info['create_date']).format('yyyy/MM/dd HH:mm:ss');
						return log_info;
					})
					console.log("lustre log ==>",data);
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
		},
		
		/**
		 * 서버에서 로그내용 불려오기
		 */
		viewLog : function(row_key,callback) {
			$.ajax({
				url: contextPath+'/api/v1/lustre/viewLog',
				type: "POST",
				data: {
					row_key : row_key
				}
			}).then(
				function ( data, textStatus, jqXHR) {
					data = _.map(data,function(log_info){
						log_info['is_view'] = operationsRunning.filter_log[log_info.log_type];
						log_info['create_date'] = new Date(log_info['create_date']).format('yyyy-MM-dd HH:mm:ss');
						return log_info;
					})
					
					console.log("lustre log viewLog ==>",data);
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
		},
		
		/**
		 * 해당 로그 출력하기
		 */
		showViewLog : function(row_key) {
			var target = '.bd-managerque-modal-lg';
			$('.console-output').empty();
			operationsRunning.viewLog(row_key,function(log_data){
				templete_html($(target)[0],'viewLog',log_data,operationsRunning.listener);
				var max_num_obj = _.max(log_data,'num');
				operationsRunning.max_num = max_num_obj ? max_num_obj['num'] : 0;
				
				// 0.5초마다 로그내용 갱신
				operationsRunning.setInterval_jobLog = setInterval( function () {
					operationsRunning.reloadLogData(operationsRunning.row_key,operationsRunning.max_num);
				}, 500 );
				
				
			});
		},
		
		/**
		 * max_num 다음으로 로그내용을 불려오는 메서드
		 */
		reloadLogData : function(row_key,max_num) {
			$.ajax({
				url: contextPath+'/api/v1/lustre/viewLastLogLine',
				type: "POST",
				data: {
					row_key : row_key,
					num : max_num
				}
			}).then(
				function ( data, textStatus, jqXHR) {
					if(data.length > 0){
						var max_num_obj = _.max(data,'num');
						operationsRunning.max_num = max_num_obj ? max_num_obj['num'] : 0;
						var target = ".bd-managerque-modal-lg .console-output";
						var html = '';
						for (var i = 0; i < data.length; i++) {
							var tmpData= data[i];
							var log_type = '';
								
							switch (tmpData['log_type']) {
								case 'command':	
									log_type = 'CMD';
									break;
								case 'info':	
									log_type = 'INFO';
									break;
								case 'error':	
									log_type = 'ERROR';
									break;
								default:
									break;
							}
							
							var str=""
							for(var j=0;j<length-log_type.length;j++){
							  str=str+"&#160;";
							}
							log_type=log_type+str;
							
							var creat_date = tmpData['create_date'] ? tmpData['create_date'].substr(0,19) : '';
							var hide_class = operationsRunning.filter_log[tmpData['log_type']] ? "":"d-none";
							
							html += '<div class="'+tmpData['log_type']+' '+hide_class+'+">'
								+'<b>'+creat_date+'</b>'
								+'<b>['+log_type+']</b>'
								+'&nbsp;'
								+tmpData['data']
								+'</div>'
						}
						$(target).append(html);
						$(target).scrollTop($(target).prop("scrollHeight"));
						operationsRunning.listener();
					}else{
//						clearInterval(operationsRunning.setInterval_jobLog);
//						operationsRunning.setInterval_jobLog = null;
					}
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
		},
		
		
		/**
		 * 로그리스트 출력
		 */
		showOperationsList : function(target) {
			operationsRunning.viewLogList(function(data) {
				templete_html($(target)[0],'OperationsList',data,operationsRunning.listener);
			});
			
		},
		
		/**
		 * 이벤트 정리
		 */
		listener : function() {
			// je.kim bootstrap multiple modal
			// https://stackoverflow.com/questions/19305821/multiple-modals-overlay
			$(document).on('show.bs.modal', '.modal', function (event) {
	            var zIndex = 1040 + (10 * $('.modal:visible').length);
	            $(this).css('z-index', zIndex);
	            setTimeout(function() {
	                $('.modal-backdrop').not('.modal-stack').css('z-index', zIndex - 1).addClass('modal-stack');
	            }, 0);
	        });
			
			$('.operationsRunning').off('click').on('click',function(){
				//var target = $(this).data('target');
				var target = '.bd-managerque-modal-lg';
				operationsRunning.showOperationsList(target);
				$(target).modal('show');
			});
			
			// 체크박스 변경시 이벤트 처리
			$('form.filter [name=log-filter-target]').off('change').on('change',function(e){
				// 필터링할 타입
				var filtering_target = $(this).val();
				
				// 체크박스 선택시
				if($(this).is(':checked')){
					operationsRunning.filter_log[filtering_target] = true;
					$('#view-logs .console-output ' + '.'+filtering_target).show();
				// 체크박스 해제시
				}else{
					operationsRunning.filter_log[filtering_target] = false;
					$('#view-logs .console-output ' + '.'+filtering_target).hide();
				}
			});
			
			$('.view-log').off('click').on('click',function(){
				try {
					clearInterval(operationsRunning.setInterval_jobLog);
					operationsRunning.setInterval_jobLog = null;
				} catch (e) {
					
				}
				
				
				
				var row_key = $(this).data('rowkey')
				operationsRunning.row_key = row_key
				operationsRunning.showViewLog(row_key);
			});
			
			$('.bd-managerque-modal-lg').on('hide.bs.modal', function (e) {
				clearInterval(operationsRunning.setInterval_jobLog);
				operationsRunning.setInterval_jobLog = null;
			});
			
//			$('.bd-managerque-modal-lg [data-dismiss=modal]').off('click').on('click',function() {
//				clearInterval(operationsRunning.setInterval_jobLog);
//				operationsRunning.setInterval_jobLog = null;
//			})
		}
}