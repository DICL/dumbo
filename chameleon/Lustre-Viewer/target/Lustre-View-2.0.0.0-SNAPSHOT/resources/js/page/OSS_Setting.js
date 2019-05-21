$(function(){
	OSS_Setting.init();
});
var OSS_Setting = {
		oss_nodes : null,
		/**
		 * 시작시 동작되는 메서드 
		 */
		init : function() {
			showLoading('get lustre information')
			
			OSS_Setting.syncLustreTable(function() {
				OSS_Setting.get_lustre_nodes(
						function(node_list, textStatus, jqXHR) {
							OSS_Setting.oss_nodes = node_list;
							var result = {};
							
							
							OSS_Setting.get_oss_network(function(oss_networks){
								
								result = _.map(node_list,function(node_info){
									//node_info.network_list = oss_networks[node_info['host_name']]
									
									var temp_oss_network_list = _.map(oss_networks[node_info['host_name']],function(item){return {name : item.split(':')[0], type : (item.split(':').length > 1) ? item.split(':')[1] : ''} })
									node_info.network_list = temp_oss_network_list
									
									return node_info;
								})
								
								OSS_Setting.get_oss_disk(function(oss_disks){
									var disk_total_count = _.reduce(oss_disks, function(memo, num){ return memo + num.length; }, 0);
									OSS_Setting['disk_total_count'] = disk_total_count;
									OSS_Setting['OST_info']={}
									for (var i = 0; i < disk_total_count; i++) {
										var temp_ost_name = "OST"+(i);
										OSS_Setting['OST_info'][temp_ost_name]=null;
									}
									
									var result = {};
									result = _.map(node_list,function(node_info){
										node_info.disk_info_list = oss_disks[node_info['host_name']]
										node_info.disk_total_count = disk_total_count;
										return node_info;
									})
									OSS_Setting.grid_oss_config_page(result,OSS_Setting.listener)
								})
							});
							
							
						}
				)
			});
			
			
		},
		
		/**
		 * OSS 노드들을 읽어오는 메서드
		 */
		get_lustre_nodes : function(callback) {
			changeLoadingText("read lustre tables");
			$.ajax({
				url: contextPath+'/api/v1/ambari/getLustreNodes',
				type: "POST",
				data: {
					node_type: 'OSS'
					,file_system_num : $('#fs_num').val() !== '' ? $('#fs_num').val() : null
				}
			}).then(
				// function ( data, textStatus, jqXHR) {}
				callback,
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
				
			);
		},
		
		/**
		 * OSS 노드들의 네트워크 정보들을 읽어오는 메서드
		 */
		get_oss_network : function(callback) {
			changeLoadingText("read lustre networks");
			$.ajax({
				url: contextPath+'/api/v1/lustre/getNetWorkDevice',
				type: "POST",
				data: {
					'node_type' : 'OSS',
				}
			}).then(function (data, textStatus, jqXHR) {
				console.log("get_oss_network ==>",data, textStatus, jqXHR)
				callback(data);
			},function(jqXHR,textStatus,errorThrown) {
				console.error(jqXHR,textStatus,errorThrown);
			})
			
			
		},
		
		
		/**
		 * shkim 20181215
		 * 암바리를 통하여 얻은 정보를 디비에 적재하는 메서드
		 */
		syncLustreTable : function(callback) {
			changeLoadingText("Get Ambari Server");
			$.ajax({
				url: contextPath+'/api/v1/ambari/syncLustreTable',
				type: "GET",
				data: {
					num : $('#fs_num').val(),
				}
			}).then(
				function ( data, textStatus, jqXHR) {
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
		
		},
		
		/**
		 * OSS 노드들의 디스크정보들을 읽어오는 메서드
		 */
		get_oss_disk : function(callback) {
			changeLoadingText("read lustre node disks");
			$.ajax({
				url: contextPath+'/api/v1/lustre/getDiskDevice',
				type: "POST",
				data: {
					'node_type' : 'OSS',
					'file_system_num' : $('#fs_num').val(),
				}
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("get_oss_disk ==>",data, textStatus, jqXHR)
					callback(data)
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
			
		},
		
		
		/**
		 * OSS 노드들의 디스크정보들을 읽어오는 메서드
		 */
		new_get_oss_disk : function(callback) {
			changeLoadingText("read lustre node disks");
			$.ajax({
				url: contextPath+'/api/v1/lustre/new_getDiskDevice',
				type: "POST",
				data: {
					'node_type' : 'OSS',
					'file_system_num' : $('#fs_num').val(),
				}
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("new_get_oss_disk ==>",data, textStatus, jqXHR)
					callback(data)
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
		},
		
		/**
		 * 템블릿 정보를 읽어와서 화면에 그리는 메서드
		 */
		grid_oss_config_page : function(data,callback) {
			showLoading('grid html...')
			console.log("get_lustre_nodes ==>",data)
			// 초기화면 숨기기
			$('#oss_page').hide();
			target = document.getElementById("oss_page")
			templete_html(target,"OSS_Setting",data,function(){
				$('#myTab li:first-child a').tab('show')
				
				OSS_Setting.new_get_oss_disk(function(new_oss_disks){
					_.each(new_oss_disks,function(disklist,hostname){
						var tmphtml = '';
						_.each(disklist,function(diskinfo){
							tmphtml += '<option value="'+ diskinfo.name +'" size="' +diskinfo.size+ '">';
							tmphtml += diskinfo.name;
							tmphtml += '</option>';
						});
						_.each($('#fs-view .disk-name[host-name='+hostname+']'),function(tem_obj){
							$(tem_obj).html(tmphtml);
						})
					});
					
					
					_.each(data,function(node_info){
						var i = 0
						// 디비에 들어있는 디스크 정보들을 돌면서 변경
						_.each(node_info['disk_list'],function(disk_info){
							//  OST 번호 수정
							$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-ost').val(disk_info['index']);
							// 수정못하게 disabled
							$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-ost').prop('disabled',true);
							
							if(typeof _.find($('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-diskname option'),function(obj){ return $(obj).val() === disk_info['disk_name'] }) === 'undefined' ){
								$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-diskname').append('<option value="'+ disk_info['disk_name'] +'">' + disk_info['disk_name'] + '</option>');
								$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-size').text(disk_info['disk_size'] );
							}
							
							// 디스크 디바이스명 수정
							$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-diskname').val(disk_info['disk_name']);
							// 수정못하게 disabled
							$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-diskname').prop('disabled',true);
							
							$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-is-activate').prop('disabled',false);
							$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-remove').prop('disabled',false);
							
							if(disk_info['is_activate'] == true){
								// activate 버튼 비활성화
								$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-deactivate').prop('disabled',false);
								// deactivate 버튼 활성화
								$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-activate').prop('disabled',true);
								$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-is-activate').prop('checked', true);
							}else{
								// activate  버튼 비활성화
								$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-activate').prop('disabled',false);
								// deactivate 버튼 활성화
								$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-deactivate').prop('disabled',true);
							}
							
							
							
							// 사이즈 표시
							$('#fs-view #OSS-'+node_info['host_name']+'-'+i+'-size').text( $('#OSS-'+node_info['host_name']+'-'+i+'-diskname option:selected').attr('size') );
							// OST정보들을 임시 변수에 저장
							OSS_Setting.OST_info['OST'+disk_info['index']] = disk_info['disk_name'];
							
							i ++;
						})
						
						// 네트워크 디바이스가 설정되어 있을경우 적용및 수정 못하게 처리
						if(node_info['network_device'] !== null && typeof node_info['network_device'] !== 'undefined'){
							$('#'+node_info['host_name']+'-network').val(node_info['network_device']);
							//$('#'+node_info['host_name']+'-network').prop('disabled',true);
						}
						// 190311 je.kim 네트워크 수정이 가능하도록 수정
						if(node_info['network_device'] === 'lo' || node_info['network_device'] === null || typeof node_info['network_device']  === 'undefined'){
							$('#'+node_info['host_name']+'-network').prop('disabled',false);
						}
						
					})
					
					$('#oss_page [data-toggle=toggle]').bootstrapToggle({
						on : 'Activate',
						off : 'Deactivate',
						width : 128,
						height: 36
					});
					hideLoading();
				});

				
				
//				$('[data-toggle=toggle]').bootstrapSwitch({
//					onText : 'Activate',
//					offText : 'Deactivate',
//				});
				
//				$('#oss_page [data-toggle=toggle]').bootstrapToggle({
//					on : 'Activate',
//					off : 'Deactivate',
//					width : 128,
//					height: 36
//				});
				
				// class 이름이 ost-name (select box) 을 순차적으로 탐색하면서
				_.each($('#fs-view .ost-name'),function(item){
					// select box 내부에 option 태그을 순차적으로 탐색
					_.each($(item).children(),function(option){
						// 만약 value 값이 none 가 아닐경우
						if($(option).val() !== 'none'){
							// 이름 지정
							var ost_name = 'OST' + $(option).val();
							// 만약 저장된 OST 정보에 등록 되어있다면 disabled 처리 (OST중복방지)
							if(OSS_Setting.OST_info[ost_name] != null && $('option:selected', item).attr('ost') != ost_name){
								$(option).prop( "disabled", true );
							}else{
								$(option).prop( "disabled", false );
							}
						}
					})
				});
				
//				hideLoading();
				// 초기화면 보이기
				$('#oss_page').show();
				callback();
			});
		},
		
		/**
		 * OST 셀렉트 를 변경시 동작되는 메서드
		 */
		set_ost_item: function() {
			var tar_get = 'ost-name';
			_.each(OSS_Setting.OST_info,function(item,key){
				OSS_Setting.OST_info[key] = null;
			})
			
			_.each($('#fs-view .'+tar_get),function(item){
				
				var value = $(item).val();
				var host_name = $(item).attr('host-name');
				var index = $(item).attr('index');
				//console.log(item)
				var disk_name = $('#fs-view  #OSS-'+host_name+'-'+index+'-diskname').val();
				
				if(value !== 'none'){
					OSS_Setting.OST_info['OST'+value] = disk_name;
				}
			});
//			console.log(OSS_Setting.OST_info)
			
			_.each($('#fs-view .'+tar_get),function(item){
				_.each($(item).children(),function(option){
					if($(option).val() !== 'none'){
						var ost_name = 'OST' + $(option).val();
						if(OSS_Setting.OST_info[ost_name] != null && $('option:selected', item).attr('ost') != ost_name){
							$(option).prop( "disabled", true );
						}else{
							$(option).prop( "disabled", false );
						}
					}
				})
			});
		},
		
		send_oss_info : function() {
			var send_node = {};
			
			showLoading('set OSS system');
			
			_.each(OSS_Setting.oss_nodes,function(item){
				send_node[item['host_name']] = {};
				send_node[item['host_name']]['disk_list'] = [];
				send_node[item['host_name']]['host_name'] = item['host_name'];
				send_node[item['host_name']]['network_device'] = $('#'+item['host_name']+'-network').val();
				send_node[item['host_name']]['node_type'] = 'OSS';
				send_node[item['host_name']]['index'] = item['index']
			})
			
			_.each($('.ost-name'),function(ost_obj){
				if($(ost_obj).val() !== 'none' && $(ost_obj).prop('disabled') === false){
					var host_name = $(ost_obj).attr('host-name');
					var select_index = $(ost_obj).attr('index');
					var disk_index = $(ost_obj).val();
					var target_disk_select_id = '#OSS-'+host_name+'-'+select_index+'-diskname';
					var disk_name = $(target_disk_select_id).val();
					console.log(target_disk_select_id)
					
					var tmp = {
							disk_type : 'OST',
							index : disk_index,
							disk_name : disk_name,
							disk_size : $('' + target_disk_select_id + ' option:selected').attr('size'),
					}
					send_node[host_name]['disk_list'].push(tmp)
				}
			});
			
			//console.log(send_node)
			// object -> array 로 전환
			var tem_list = _.map(send_node,function(item){
				item.file_system_num = $('#fs_num').val();
				return item;
			})
			var data = {};
			data['list'] = tem_list;
			data['file_system_num'] = $('#fs_num').val();
			//console.log(data)
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/ossAddDisk_fs',
				type: "POST",
				data: JSON.stringify( data ),
				dataType: "json",
			    contentType : 'application/json'
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("send_oss_disk ==>",data, textStatus, jqXHR)
					alert('OSS Setting Complete');
					hideLoading();
					OSS_Setting.init(); // 새로고침
//					operationsRunning.showViewLog(data[0]['data']);
//					var target = '.bd-managerque-modal-lg';
//					$(target).modal('show');
					
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
			
			
			
		},
		
		/**
		 * deactivate 버튼클릭시 함수
		 */
		deactivate : function(host_name,index) {
			// disk name
			var disk_name = $('#OSS-'+host_name+'-'+index+'-diskname').val();
			// ost number
			var ost_index = $('#OSS-'+host_name+'-'+index+'-ost').val();
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/ostDeactivate',
				type: "POST",
				data: {
					host_name : host_name,
					disk_name : disk_name,
					index : ost_index,
					disk_type : 'OST',
					file_system_num : $('#fs_num').val(),
				},
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("deactivate ==>",data, textStatus, jqXHR)
					
					$('#OSS-'+host_name+'-'+index+'-activate').prop( "disabled", false );
					$('#OSS-'+host_name+'-'+index+'-deactivate').prop( "disabled", true );
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
		},
		
		activate : function(host_name,index) {
			// disk name
			var disk_name = $('#OSS-'+host_name+'-'+index+'-diskname').val();
			// ost number
			var ost_index = $('#OSS-'+host_name+'-'+index+'-ost').val();
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/ostActivate',
				type: "POST",
				data: {
					host_name : host_name,
					disk_name : disk_name,
					index : ost_index,
					disk_type : 'OST',
					file_system_num : $('#fs_num').val(),
				},
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("deactivate ==>",data, textStatus, jqXHR)
					
					$('#OSS-'+host_name+'-'+index+'-activate').prop( "disabled", true );
					$('#OSS-'+host_name+'-'+index+'-deactivate').prop( "disabled", false );
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
		},
		
		remove_ost : function(host_name,index) {
			// disk name
			var disk_name = $('#OSS-'+host_name+'-'+index+'-diskname').val();
			// ost number
			var ost_index = $('#OSS-'+host_name+'-'+index+'-ost').val();
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/removeOST',
				type: "POST",
				data: {
					host_name : host_name,
					disk_name : disk_name,
					index : ost_index,
					disk_type : 'OST',
					file_system_num : $('#fs_num').val(),
				},
				beforeSend : function( jqXHR , settings) {
					showLoading('remove ost');
				},
				complete : function(jqXHR,textStatus) {
					
				},
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("remove ==>",data, textStatus, jqXHR)
					$('#OSS-'+host_name+'-'+index+'-ost').prop("disabled", false);
					$('#OSS-'+host_name+'-'+index+'-diskname').prop("disabled", false);
					$('#OSS-'+host_name+'-'+index+'-remove').prop("disabled", true);
					$('#OSS-'+host_name+'-'+index+'-is-activate').bootstrapSwitch('toggleDisabled', false, false);
					alert('OST Remove Complete');
					OSS_Setting.init(); // 새로고침
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
		},
		
		
		/**
		 * 이벤트 함수
		 */
		listener : function() {
			$('.disk-name').off('change').on('change',function(){
				var index = $(this).attr('index')
				var host_name = $(this).attr('host-name')
				
				$('#OSS-'+host_name+'-'+index+'-size').text($('option:selected', this).attr('size'));
			});
			
			$('.ost-name').off('change').on('change',function(){
				var index = $(this).attr('index')
				var host_name = $(this).attr('host-name')
				
//				console.log('OSS-'+host_name+'-'+index+'-diskname')
				
				if($(this).val() === 'none'){
					$('#OSS-'+host_name+'-'+index+'-diskname').prop( "disabled", true );
				}else{
					$('#OSS-'+host_name+'-'+index+'-diskname').prop( "disabled", false );
				}
				
				OSS_Setting.set_ost_item();
				
			});
			
//			$('.activate-ost').off('click').on('click',function(){
//				var host_name = $(this).attr('host-name');
//				var index = $(this).attr('index');
//				OSS_Setting.activate(host_name,index);
//			});
//			
//			$('.deactivate-ost').off('click').on('click',function(){
//				var host_name = $(this).attr('host-name');
//				var index = $(this).attr('index');
//				OSS_Setting.deactivate(host_name,index);
//			});
			
//			$('.set-activate-toggle').on('switchChange.bootstrapSwitch', function(event, state) {
////				console.log(this); // DOM element
////				console.log(event); // jQuery event
////				console.log(state); // true | false
//				
//				var host_name = $(this).attr('host-name');
//				var index = $(this).attr('index');
//				
//				if(state === true){ //activate
//					OSS_Setting.activate(host_name,index);
//				}else{ // deactivate
//					OSS_Setting.deactivate(host_name,index);
//				}
//			})
			
			
			$('.set-activate-toggle').change(function() {
				
				var host_name = $(this).attr('host-name');
				var index = $(this).attr('index');
				var state =$(this).prop('checked');
				if(state === true){ //activate
					OSS_Setting.activate(host_name,index);
				}else{ // deactivate
					OSS_Setting.deactivate(host_name,index);
				}
			})
			
			
			$('.remove-ost').off('click').on('click',function(){
				var host_name = $(this).attr('host-name');
				var index = $(this).attr('index');
				var ost_index = $("#OSS-"+host_name+"-"+index+"-ost").val();
				if(confirm("Are you sure remove "+host_name+" OST"+ost_index+" ??") ){
					OSS_Setting.remove_ost(host_name,index);
				}
			});
			
			$('#oss-submit').off('click').on('click',function(){
				OSS_Setting.send_oss_info();
			});
			
			$('#oss-reset').off('click').on('click',function(){
				OSS_Setting.init();
			});
			
			
			$('.ost-network').off('change').on('change',function(){
				//console.table($(this))
				var host_name = $(this).data('host-name');
				var network_type = $('option:selected', this).data('type');
				var network_name = $(this).val();
				var is_tcp = network_type === 'infiniband' ? 'o2ib1' : 'tcp';
				var result = 'options lnet networks="'+is_tcp+'('+network_name+')"';
				$('#'+host_name+'-network_option').val(result);
			});
		}
		
}