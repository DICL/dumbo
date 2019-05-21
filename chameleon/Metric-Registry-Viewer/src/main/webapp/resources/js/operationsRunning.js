$(function(){
	operationsRunning.listener();
});

var operationsRunning = {
		setInterval_jobLog : null,
		max_num:0,
		row_key:null,
		viewLogList : function(callback) {
			$.ajax({
				url: contextPath+'/api/v1/lustre/getLogList',
				type: "POST",
				data: {}
			}).then(
				function ( data, textStatus, jqXHR) {
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
		
		viewLog : function(row_key,callback) {
			$.ajax({
				url: contextPath+'/api/v1/lustre/viewLog',
				type: "POST",
				data: {
					row_key : row_key
				}
			}).then(
				function ( data, textStatus, jqXHR) {
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
		
		showViewLog : function(row_key) {
			var target = '.bd-managerque-modal-lg';
			$('.console-output').empty();
			operationsRunning.viewLog(row_key,function(log_data){
				templete_html($(target)[0],'viewLog',log_data,operationsRunning.listener);
				var max_num_obj = _.max(log_data,'num');
				operationsRunning.max_num = max_num_obj ? max_num_obj['num'] : 0;
				
				
				operationsRunning.setInterval_jobLog = setInterval( function () {
					operationsRunning.reloadLogData(operationsRunning.row_key,operationsRunning.max_num);
				}, 500 );
				
				
			});
		},
		
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
							html += '<div class="'+tmpData['log_type']+'">'
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
		
		
		showOperationsList : function(target) {
			operationsRunning.viewLogList(function(data) {
				templete_html($(target)[0],'OperationsList',data,operationsRunning.listener);
			});
			
		},
		
		listener : function() {
			$('.operationsRunning').off('click').on('click',function(){
				//var target = $(this).data('target');
				var target = '.bd-managerque-modal-lg';
				operationsRunning.showOperationsList(target);
				$(target).modal('show');
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