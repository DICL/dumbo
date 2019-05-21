$(function(){
	LNET_Setting.init();
});
var LNET_Setting = {
		node_list : null,
		/**
		 * 최초 시작시 동작하는 메서드
		 */
		init : function() {
			// 로딩화면 출력
			showLoading('start lustre information')
			// api 호출후에
			LNET_Setting.get_lustre_conf(function ( conf_data, textStatus, jqXHR) {
				LNET_Setting.node_list = conf_data;
				console.log("get_lustre_confs ==>",conf_data)
				// 결과물을 이용하여 화면에 출력
				
				LNET_Setting.get_network(function(network_data) {
					var result = _.map(conf_data,function(conf){
						conf['network_list'] = _.map(network_data[conf['host_name']] ,function(item){return {name : item.split(':')[0], type : (item.split(':').length > 1) ? item.split(':')[1] : ''} } );
						return conf;
					});
					
					LNET_Setting.grid_lustre_conf_page(result,function(){
						LNET_Setting.listener();
					})
				});
				
				IndexPage.get_file_system_list().then(
					function(data, textStatus, jqXHR){
						templete_html($('#fs-list')[0],"index_fs_list",data,function(data){
							$('[data-num='+  $('#fs_num').val() +']').addClass('active');
							$('[data-fstoggle=toggle]').bootstrapToggle({
								on : 'mount',
								off : 'umount',
								size : 'small',
							});
							IndexPage.listener();
						});
					},
					IndexPage.errorAjax
				);
			});
		},
		
		/**
		 * 노드들의 네트워크 정보들을 읽어오는 메서드
		 */
		get_network : function(callback) {
			changeLoadingText("read lustre networks");
			$.ajax({
				url: contextPath+'/api/v1/lustre/getNetWorkDevice',
				type: "POST",
				data: {
				}
			}).then(function (data, textStatus, jqXHR) {
				console.log("get_lustre_network ==>",data, textStatus, jqXHR)
				callback(data);
			},function(jqXHR,textStatus,errorThrown) {
				console.error(jqXHR,textStatus,errorThrown);
			})
		},
		
		/**
		 * lustre.conf 읽어오는 api 호출
		 */
		get_lustre_conf : function(callback) {
			// 로딩텍스트 변경
			changeLoadingText("read lustre.conf files");
			// ajax 호출
			$.ajax({
				url: contextPath+'/api/v1/lustre/readLustreConf',
				type: "POST",
				data: {
					'file_system_num' : $('#fs_num').val(),
					// 190311 je.kim 클라이언트 노드만 표시
					'node_type' : 'CLIENT',
				}
			}).then(
				// function ( data, textStatus, jqXHR) {}
				// 성공시 콜백함수 호출
				callback,
				// 실패시 에러로그 출력
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
				
			);
		},
		
		/**
		 * api로 받아온 결과를 화면에 그리기
		 */
		grid_lustre_conf_page : function(conf_list,callback) {
			showLoading('grid html...')
			console.log(conf_list)
			target = document.getElementById("LNET_Setting_List")
			templete_html(target,"LNET_Setting",conf_list,function(){
				_.each(conf_list,function(item){
					$('#'+item.host_name+'-network').val(item.network_device)
				})
				hideLoading();
				callback();
			});
		},
		
		
		network_start : function(host_name,node_type) {
			var conf_file_data = $('#'+host_name+'-conf').val();
			// ajax 호출
			//console.log(host_name,node_type,conf_file_data)
			$.ajax({
				url: contextPath+'/api/v1/lustre/networkStart_fs',
				type: "POST",
				data: {
					host_name : host_name,
					node_type : node_type,
					conf_file_data : conf_file_data,
					file_system_num : $('#fs_num').val(),
				},
				// 서버에 보내기전에 processing.. 문구 출력
				beforeSend : function( jqXHR , settings) {
					$('.lnet-status[data-host-name='+host_name+']').text('processing..');
					$('#fs-view .network-start').prop('disabled',true);
					$('#fs-view .network-stop').prop('disabled',true);
					$('#fs-view #start-all').prop('disabled',true);
					$('#fs-view #stop-all').prop('disabled',true);
				},
				complete : function(jqXHR,textStatus) {
					$('.lnet-status[data-host-name='+host_name+']').text('done.');
					$('#fs-view .network-start').prop('disabled',false);
					$('#fs-view .network-stop').prop('disabled',false);
					$('#fs-view #start-all').prop('disabled',false);
					$('#fs-view #stop-all').prop('disabled',false);
				},
			}).then(
				// 성공시 호출
				function ( data, textStatus, jqXHR) {
					console.log(data)
					alert('start LNET Setting');
					LNET_Setting.init();
				},
				// 실패시 출력
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
				
			);
		},
		
		network_stop : function(host_name,node_type) {
			$.ajax({
				url: contextPath+'/api/v1/lustre/networkStop_fs',
				type: "POST",
				data: {
					host_name : host_name,
					node_type : node_type,
					file_system_num : $('#fs_num').val(),
				},
				// 서버에 보내기전에 processing.. 문구 출력
				beforeSend : function( jqXHR , settings) {
					$('.lnet-status[data-host-name='+host_name+']').text('processing..');
					$('#fs-view .network-start').prop('disabled',true);
					$('#fs-view .network-stop').prop('disabled',true);
					$('#fs-view #start-all').prop('disabled',true);
					$('#fs-view #stop-all').prop('disabled',true);
				},
				complete : function(jqXHR,textStatus) {
					$('.lnet-status[data-host-name='+host_name+']').text('done.');
					$('#fs-view .network-start').prop('disabled',false);
					$('#fs-view .network-stop').prop('disabled',false);
					$('#fs-view #start-all').prop('disabled',false);
					$('#fs-view #stop-all').prop('disabled',false);
				},
			}).then(
				// 성공시 호출
				function ( data, textStatus, jqXHR) {
					console.log(data)
					alert('stop LNET Setting');
					LNET_Setting.init();
				},
				// 실패시 출력
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
				
			);
		},
		
		start_all : function() {
			if(LNET_Setting.node_list.length == 0) return false;
			var data = _.map(LNET_Setting.node_list,function(item){
				item.data = $('#'+item.host_name+'-conf').val()
				item.file_system_num = $('#fs_num').val()
				return item;
			})
			$.ajax({
				url: contextPath+'/api/v1/lustre/networkAllStart_fs',
				type: "POST",
				data: JSON.stringify( data ),
				dataType: "json",
			    contentType : 'application/json',
			    // 서버에 보내기전에 processing.. 문구 출력
				beforeSend : function( jqXHR , settings) {
					_.each($('.lnet-status'),function(object){
						$(object).text('processing..');
					});
					$('#fs-view .network-start').prop('disabled',true);
					$('#fs-view .network-stop').prop('disabled',true);
					$('#fs-view #start-all').prop('disabled',true);
					$('#fs-view #stop-all').prop('disabled',true);
				},
				complete : function(jqXHR,textStatus) {
					_.each($('.lnet-status'),function(object){
						$(object).text('done.');
					});
					$('#fs-view .network-start').prop('disabled',false);
					$('#fs-view .network-stop').prop('disabled',false);
					$('#fs-view #start-all').prop('disabled',false);
					$('#fs-view #stop-all').prop('disabled',false);
				},
			}).then(
				// 성공시 호출
				function ( data, textStatus, jqXHR) {
					console.log(data)
					alert('start All LNET Setting');
					LNET_Setting.init();
				},
				// 실패시 출력
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
		},
		
		
		stop_all : function() {
			if(LNET_Setting.node_list.length == 0) return false;
			var data = LNET_Setting.node_list;
			$.ajax({
				url: contextPath+'/api/v1/lustre/networkAllStop_fs',
				type: "POST",
				data: JSON.stringify( data ),
				dataType: "json",
			    contentType : 'application/json',
			 // 서버에 보내기전에 processing.. 문구 출력
				beforeSend : function( jqXHR , settings) {
					_.each($('.lnet-status'),function(object){
						$(object).text('processing..');
					});
					$('#fs-view .network-start').prop('disabled',true);
					$('#fs-view .network-stop').prop('disabled',true);
					$('#fs-view #start-all').prop('disabled',true);
					$('#fs-view #stop-all').prop('disabled',true);
				},
				complete : function(jqXHR,textStatus) {
					_.each($('.lnet-status'),function(object){
						$(object).text('done.');
					});
					$('#fs-view .network-start').prop('disabled',false);
					$('#fs-view .network-stop').prop('disabled',false);
					$('#fs-view #start-all').prop('disabled',false);
					$('#fs-view #stop-all').prop('disabled',false);
				},
			}).then(
				// 성공시 호출
				function ( data, textStatus, jqXHR) {
					console.log(data)
					alert('stop All LNET Setting');
					LNET_Setting.init();
				},
				// 실패시 출력
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
		},
		
		
		/**
		 * 이벤트 함수
		 */
		listener : function() {
			$('.network-start').off('click').on('click',function(e){
				var host_name = $(this).data('host-name');
				var node_type = $(this).data('node-type');
				LNET_Setting.network_start(host_name,node_type);
			});
			$('.network-stop').off('click').on('click',function(e){
				var host_name = $(this).data('host-name');
				var node_type = $(this).data('node-type');
				LNET_Setting.network_stop(host_name,node_type);
			});
			$('#start-all').off('click').on('click',function(e){
				LNET_Setting.start_all();
			});
			$('#stop-all').off('click').on('click',function(e){
				LNET_Setting.stop_all();
			});
			
			$('.lustre-network').off('change').on('change',function(e){
				var host_name = $(this).data('host-name');
				var network_type = $('option:selected', this).data('type');
				var network_name = $(this).val();
				var is_tcp = network_type === 'infiniband' ? 'o2ib1' : 'tcp';
				var result = 'options lnet networks="'+is_tcp+'('+network_name+')"';
				$('#'+host_name+'-conf').val(result);
			});
		}
}