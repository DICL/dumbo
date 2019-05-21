$(function(){
	Client_Setting.init();
});

/**
 *  /#/main/views/Lustre_View/1.0.0/Lustre_View
 */
var Client_Setting = {
		
		/**
		 *  client host 정보
		 */
		host_list : null,
		
		/**
		 * 최초 시작 메서드
		 * @returns
		 */
		init : function() {
			showLoading('get lustre information')
//			hideLoading();
//			Client_Setting.listener();
			Client_Setting.getClientList().then(
				Client_Setting.gridClient
				,Client_Setting.errorAjax
			);
			
			// 190411 je.kim 클라이언트 세팅후에는 리플래쉬 처리
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
		},
		
		/**
		 * client node 정보 가져오기
		 */
		getClientList : function() {
			return $.ajax({
				url: contextPath+'/api/v1/ambari/getLustreNodes',
				type: "POST",
				data: {
					node_type : 'CLIENT',
					'file_system_num' : $('#fs_num').val(),
				}
			})
		},
		
		/**
		 * CLIENT 노드들의 네트워크 정보들을 읽어오는 메서드
		 */
		getClientNetworkList : function(callback) {
			changeLoadingText("read lustre networks");
			$.ajax({
				url: contextPath+'/api/v1/lustre/getNetWorkDevice',
				type: "POST",
				data: {
					'node_type' : 'CLIENT',
					'file_system_num' : $('#fs_num').val(),
				}
			}).then(function (data, textStatus, jqXHR) {
				console.log("get_oss_network ==>",data, textStatus, jqXHR)
				callback(data);
			},function(jqXHR,textStatus,errorThrown) {
				console.error(jqXHR,textStatus,errorThrown);
			})
		},
		
		/**
		 * 가져온 client 정보를 토대로 화면에 그리는 메서드
		 */
		gridClient : function( data, textStatus, jqXHR) {
			console.log('gridClient ==>',data)
			// 호스트 리스트 들을 변수에 저장
			Client_Setting.host_list = _.map(data,function(item){return item.host_name});
			// 화면에 그릴 dom 겍체
			target = document.getElementById("client_page")
			var node_list = data;
			var mount_point = (node_list.length > 0) ? node_list[0].lustre_client_folder : '/lustre';
			
			Client_Setting.getClientNetworkList(function(network_list) {
				var result = _.map(node_list,function(node_info){
					var temp_oss_network_list = _.map(network_list[node_info['host_name']],function(item){return {name : item.split(':')[0], type : (item.split(':').length > 1) ? item.split(':')[1] : ''} })
					node_info.network_list = temp_oss_network_list
							
					return node_info;
				})
				templete_html(target,"Client_Setting",result,function(){
					_.each(node_list,function(item){
						$('#'+item.host_name+'-network').val(item.network_device);
						// 190311 je.kim 네트워크 수정이 가능하도록 수정
						if(item['network_device'] === 'lo' || item['network_device'] === null || typeof item['network_device']  === 'undefined'){
							$('#'+item.host_name+'-network').prop('disabled',false);
						}
					});
					$('#client-mount-name').val(mount_point)
					hideLoading();
					Client_Setting.listener();
				});
			})
		},
		
		/**
		 * 에러시 처리되는 메서드
		 */
		errorAjax : function(jqXHR,textStatus,errorThrown) {
			console.error(jqXHR,textStatus,errorThrown);
		},
		
		
		/**
		 * 서버에 보낼 클라이언트 정보
		 */
		clientSetting_bak : function() {
			if(!$('#client-network-name').val() || !$('#client-mount-name').val()){
				return false;
			}
			$.ajax({
				url: contextPath+'/api/v1/lustre/clientSetting',
				type: "POST",
				data: {
					client_network : $('#fs-view #client-network-name').val(),
					client_mount_point : $('#fs-view #client-mount-name').val(),
				}
			}).then(
				// function ( data, textStatus, jqXHR) {}
				function ( data, textStatus, jqXHR) {
					console.log("send_client_info ==>",data, textStatus, jqXHR)
					alert('client setting');
//					if(data.length > 0){
//						operationsRunning.showViewLog(data[0]['data']);
//						var target = '.bd-managerque-modal-lg';
//						$(target).modal('show');
//					}
					
				},
				Client_Setting.errorAjax
			);
		},
		
		
		/**
		 * 서버에 보낼 클라이언트 정보
		 */
		clientSetting : function() {
			var host_list = Client_Setting.host_list;
			showLoading('send server...');
			if(host_list.lenth <= 0){
				return false;
			}
//			for (var i = 0; i < host_list.length; i++) {
//				var host_name = host_list[i];
//				if(!$('#' +  host_name + '-network').val()){
//					alert('Please enter a network name');
//					$('#' +  host_name + '-network').focus();
//					return false;
//				}
//				if(!$('#' +  host_name + '-network_option').val()){
//					alert('Please enter a config');
//					$('#' +  host_name + '-network_option').focus();
//					return false;
//				}
//			}
			var send_data = {};
			send_data['list'] = _.map(host_list,function(host_name){
				
				if( $('#fs-view #' +  host_name + '-network').is(':disabled') ){
					var network_device = null;
				}else{
					var network_device =  $('#' +  host_name + '-network').val();
				}
				
				return {
					host_name : host_name,
					network_device : network_device,
					node_type : 'CLIENT',
					network_option : $('#' +  host_name + '-network_option').val(),
					lustre_client_folder :  $('#client-mount-name').val(),
				}
			});
			
			send_data['file_system_num'] = $('#fs_num').val();
			send_data['lustre_client_folder'] = $('#client-mount-name').val();
			
			console.log(send_data)
			
			$.ajax({
//				url: contextPath+'/api/v1/lustre/clientSetting_fs',
				url: contextPath+'/api/v1/lustre/clientMountFolderSetting',
				type: "POST",
				data: JSON.stringify( send_data ),
				dataType: "json",
			    contentType : 'application/json',
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("send_client_info ==>",data, textStatus, jqXHR)
					alert('client setting');
					hideLoading();
					Client_Setting.init();
				},
				Client_Setting.errorAjax
			);
			
		},
		
		/**
		 * 이벤트 처리 메서드 리스트
		 */
		listener : function() {
			$('#fs-view #client-apply').off('click').on('click',function(e){
				var message = "If you change the client folder, \n\nthe changes are made to the mount after the unmount. \n\nDo you want to proceed?";
				if(confirm(message)){
					Client_Setting.clientSetting();
				}
			});
			
			$('#fs-view #client-reset').off('click').on('click',function(e){
				Client_Setting.init();
			});
			
			$('#fs-view .client-network').off('change').on('change',function(e){
				var host_name = $(this).data('host-name');
				var network_type = $('option:selected', this).data('type');
				var network_name = $(this).val();
				var is_tcp = network_type === 'infiniband' ? 'o2ib1' : 'tcp';
				var result = 'options lnet networks="'+is_tcp+'('+network_name+')"';
				$('#'+host_name+'-network_option').val(result);
			});
		}
}