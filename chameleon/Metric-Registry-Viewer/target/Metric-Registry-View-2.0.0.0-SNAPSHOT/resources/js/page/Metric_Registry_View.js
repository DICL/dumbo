$(function(){
	Metric_Registry_View.init();
});
var Metric_Registry_View = {
		/**
		 * 최초 시작시 함수
		 */
		init : function() {
			showLoading('GET Metric List.....');
			
			Metric_Registry_View.checkMetricRegistryTable(
					function(){
						Metric_Registry_View.getMetricCycleTime().then(function ( data, textStatus, jqXHR) {
							if(data.status){
								$('.metric-cycle-time #cycle_time').val(data.cycle_time);
							}
						});
						
						Metric_Registry_View.getMetricList().then(
							function ( data, textStatus, jqXHR) {
								console.log("metric log ==>",data);
								var result = {
										list : data,
										contextPath : contextPath,
								}
								templete_html (document.getElementById("metric_list"),"Metric_Registry_View",result, Metric_Registry_View.listener)
								hideLoading();
							},
							function(jqXHR,textStatus,errorThrown) {
								console.error(jqXHR,textStatus,errorThrown);
							}		
						);
					}, //success
					function(){
						//$('#root').hide();
						$('.bd-managerque-modal-lg[aria-labelledby=myLargeModalLabel]').modal('show');
						hideLoading();
						$('#create-table').off('click').on('click',function(){
							Metric_Registry_View.init_table(
									function(){ alert('create table success!'); window.location.reload(); },
									function(){ alert('create table fail')}
							);
						})
					}  //false
			);
			
		},
		
		init_table : function(success_function,fail_function) {
			showLoading('Create Table.....');
			$('.bd-managerque-modal-lg[aria-labelledby=myLargeModalLabel]').css('z-index','0');
			return $.ajax({
				url: contextPath+'/api/v1/metric/initTable',
				type: "POST",
				data: {
				},
				success : function(res) {
					if( typeof res !== 'undefined' && res){
						success_function();
					}else{
						fail_function();
					}
				},
			}) 
		},
		
		
		/**
		 * 테이블 생성 테스
		 */
		checkMetricRegistryTable : function(success_function,fail_function){
			$.ajax({
				url: contextPath+'/api/v1/metric/checkMetricRegistryTable',
				type: "GET",
				data: {
				},
				success : function(res) {
					if( typeof res !== 'undefined' && res){
						success_function();
					}else{
						fail_function();
					}
				},
				
			}) 
		},
		
		/**
		 * 크론탭 주기 가져오기
		 */
		getMetricCycleTime : function() {
			return $.ajax({
				url: contextPath+'/api/v1/metric/getMetricCycleTime',
				type: "GET",
				data: {
				}
			}) 
		},
		
		/**
		 * 크론탭 주기 변경
		 */
		updateMetricCycleTime : function(cycle_time) {
			return $.ajax({
				url: contextPath+'/api/v1/metric/updateMetricCycleTime',
				type: "POST",
				data: {
					cycle_time : cycle_time
				}
			}) 
		},
		
		/**
		 * 메트릭 제거
		 */
		deleteMetric : function(num){
			return $.ajax({
				url: contextPath+'/api/v1/metric/deleteMetric',
				type: "POST",
				beforeSend : function() {
					showLoading('Delete Metric .....');
				},
				data: {
					num : num
				}
			}) 
		},
		
		/**
		 * 메트릭 리스트 가져오기
		 */
		getMetricList : function() {
			return $.ajax({
				url: contextPath+'/api/v1/metric/getMetricList',
				type: "GET",
				data: {}
			})
		},
		
		/**
		 * 이벤트 정리
		 */
		listener : function() {
			// 메트릭 주기 업데이트
			$('.metric-cycle-time #update_cycle_time').off('click').on('click',function(){
				var cycle_time = $('.metric-cycle-time #cycle_time').val();
				cycle_time = parseInt(cycle_time);
				if(confirm("Are you sure you want to update crontap cycle?")){
					Metric_Registry_View.updateMetricCycleTime(cycle_time).then(
							function ( data, textStatus, jqXHR) {
								if(data.status){
									alert("Update Cycle Time : " + cycle_time);
								}else{
									console.log(data);
								}
							}
					)
				}
			});
			// 메트릭 삭제
			$('.delete-metric').off('click').on('click',function(){
				var num = $(this).data('num');
				var name = $(this).data('name');
				if(confirm("Are you sure you want to delete it?")){
					Metric_Registry_View.deleteMetric(num).then(
							function ( data, textStatus, jqXHR) {
								if(data){
									alert("Delete Metric : " + name);
								}else{
									alert("Delete Fail");
								}
								hideLoading();
								Metric_Registry_View.init();
							},
							function(jqXHR,textStatus,errorThrown) {
								console.error(jqXHR,textStatus,errorThrown);
							}	
					);
				}
			});
		}
}