$(function(){
	MDS_Setting.init();
});

var MDS_Setting = {
		nodes : null,

		/**
		 * 시작시 동작되는 메서드 
		 */
		init : function() {
			showLoading("check tables");
			$('#mds-disk-info').text('');

			MDS_Setting.get_lustre_nodes(function(lustre_nodes) {
				
				var ajax_total_count = 2;
				var ajax_count = 0;
				
				var mds_node_info = _.filter(lustre_nodes,function(item){ return item.node_type === "MDS" });
				mds_node_info = mds_node_info.length > 0 ? mds_node_info[0] : {};
				MDS_Setting.nodes = mds_node_info;
				
				$('.mds-hostname').text(mds_node_info['host_name'])
				changeLoadingText("Read MDS Information.....");
				
				MDS_Setting.get_mds_disk(function(disk_list) {
					ajax_count ++;
					
//					if(mds_node_info.disk_list.length > 0){
//						MDS_Setting.grid_disk_information(mds_node_info.disk_list);
//						$('#mds-disk-name').hide();
//						$('#mds-disk-info').hide();
//					}else{
//						MDS_Setting.grid_disk_options(disk_list[mds_node_info.host_name]);
//					}
					MDS_Setting.grid_disk_options(disk_list[mds_node_info.host_name]);
					
					if(mds_node_info['disk_list'].length > 0){
						
						if(typeof _.find($('#mds-disk-name option'),function(obj){ return $(obj).val() === mds_node_info['disk_list'][0]['disk_name'] }) === 'undefined' ){
							$('#mds-disk-name').append('<option value="'+ mds_node_info['disk_list'][0]['disk_name'] +'">' + mds_node_info['disk_list'][0]['disk_name'] + '</option>');
							$('#mds-disk-info').text( mds_node_info['disk_list'][0]['disk_size'] );
						}
						
						$('#mds-disk-name').val(mds_node_info['disk_list'][0]['disk_name']);
						$('#mds-disk-name').prop('disabled', true);
						$('#mds-apply').prop('disabled', true);
					}
					
					if(ajax_count >= ajax_total_count){
						hideLoading();
						MDS_Setting.listener();
					}
					
				})
				
				MDS_Setting.get_mds_network(function(network_list) {
					ajax_count ++;
					//var tmp_network_list = _.map(network_list[mds_node_info.host_name],function(network_name){return network_name.split(':')[0]})
					MDS_Setting.grid_network_options(network_list[mds_node_info.host_name]);
					
					if(typeof mds_node_info['network_device'] !== 'undefined' && mds_node_info['network_device'] !== null && mds_node_info['network_device'] != ""){
						$('#mds-network-info').val(mds_node_info['network_device']);
						$('#network_option').val(mds_node_info['network_option']);
						// 수정후 주석 풀
						//$('#mds-network-info').prop('disabled', true);
					}
					
					if(ajax_count >= ajax_total_count){
						hideLoading();
						MDS_Setting.listener();
					}
				})
				
			});
		},
		
		grid_disk_information : function(disk_list) {
			var mdt_disk_info = _.filter(disk_list,function(item){ return item.disk_type === "MDT" });
			mdt_disk_info = mdt_disk_info[0];
			var html = '';
			html += '<input disabled class="form-control col-6" value="'+mdt_disk_info.disk_name+'">';
			html += '<small class="ml-5">';
			html += mdt_disk_info.disk_size;
			html += '</small>';
			html += '';
			
			$("#save-mds-disk-infomation").html(html);
			$("#save-mds-disk-infomation").show();
		},
		
		/**
		 * select 내의 option 태그를 그리는 메서드 (네트워크 정보)
		 */
		grid_network_options : function(network_list) {
			var html = "";
			html += "<option value=\"\">Select NETWORK</option>";
			_.each(network_list,function(item){
				html += "<option value=\""+item.split(':')[0]+"\" data-type=\""+item.split(':')[1]+"\">";
				html += item.split(':')[0] + ' ('+item.split(':')[1]+')';
				html += "</option>";
			})
			$("#mds-network-info").html(html);
		},
		
		
		/**
		 * select 내의 option 태그를 그리는 메서드 (디스크 정보)
		 */
		grid_disk_options : function(data_list) {
			var html = "";
			html += "<option value=\"\" size=\"\">Select DISK</option>";
			_.each(data_list,function(item){
				html += "<option value=\""+item.name+"\" size=\""+item.size+"\">";
				html += item.name;
				html += "</option>";
			})
			$("#mds-disk-name").html(html);
		},
		
		
		
		
		/**
		 * MDS 내의 네트워크 정보를 읽어오는 메서드
		 */
		get_mds_network : function(callback) {
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/getNetWorkDevice',
				type: "POST",
				data: {
					'node_type' : 'MDS',
					'index' : 0,
				}
			}).then(function (data, textStatus, jqXHR) {
				console.log("get_mds_networks ==>",data, textStatus, jqXHR)
				callback(data);
			},function(jqXHR,textStatus,errorThrown) {
				console.error(jqXHR,textStatus,errorThrown);
			})
			
			
		},
		
		/**
		 * MDS 내의 디스크 정보를 읽어오는 메서드
		 */
		get_mds_disk : function(callback) {
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/new_getDiskDevice',
				type: "POST",
				data: {
					'node_type' : 'MDS',
					'index' : 0,
					'file_system_num' : $('#fs_num').val(),
				}
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("get_mds_disk ==>",data, textStatus, jqXHR)
					callback(data)
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
			
		},
		
		
		/**
		 * 암바리를 통하여 얻은 정보를 디비에 적재하는 메서드
		 */
		syncLustreTable : function(callback) {
			changeLoadingText("Get Ambari Server");
			$.ajax({
				url: contextPath+'/api/v1/ambari/syncLustreTable',
				type: "GET",
				data: {}
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
		 * 데이터 베이스 테이블이 생성되었는지 확인하는 메서드
		 */
		check_created_tables : function(callback) {
			$.ajax({
				url: contextPath+'/api/v1/lustre/chechCreatedTables',
				type: "GET",
				data: {}
			}).then(
				function ( data, textStatus, jqXHR) {
					if(!data.status){
						  alert("테이블 생성 오류");
					  }else{
						  try {
							  callback(data);
							} catch (e) {
								console.warn(e);
							}
						  
					 }
				},
				function(jqXHR,textStatus,errorThrown) {
					console.error(jqXHR,textStatus,errorThrown);
				}
			);
			
			
		},
		
		/**
		 * 디비에 데이터가 있는지 확인하고 없으면 암바리를 통하여 디비에 적재하는 메서드
		 */
		get_lustre_nodes : function(callback) {
			$.ajax({
				url: contextPath+'/api/v1/ambari/getLustreNodes',
				type: "POST",
				data: {
					node_type : 'MDS',
					file_system_num : $('#fs_num').val() !== '' ? $('#fs_num').val() : null
				}
			}).then(
				function ( data, textStatus, jqXHR) {
					console.log("lustre nodes ==>",data);
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
		 * mds 설정값을 디비에 보내는 함수 
		 */
		send_mds_info : function() {
			var tmpdata = MDS_Setting.nodes;
			var data = {};
			
			if(!$('#mds-disk-name').val()){
				alert('select disk name');
				$('#mds-disk-name').focus();
				return false;
			}
			
			if(!$('#mds-network-info').val()){
				alert('select network name');
				$('#mds-network-info').focus();
				return false;
			}
			
			data['host_name'] = tmpdata.hostname;
			data['index'] = tmpdata.index;
			data['disk_type'] = 'MDT';
			data['node_type'] = 'MDS';
			data['index'] = 0;
			data['disk_name'] = $('#mds-disk-name').val();
			data['network_device'] = $('#mds-network-info').val();
			data['network_option'] = $('#network_option').val();
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/mdsAddDisk',
				type: "POST",
				data: data
			}).then(
					function ( data, textStatus, jqXHR) {
//						alert('MDS Setting');
						
//						if(data.status){
//							var row_key = data.data;
//							operationsRunning.showViewLog(row_key);
//							var target = '.bd-managerque-modal-lg';
//							$(target).modal('show');
//						}
						alert('MDS Setting')
						
						
						$('#mds-disk-name').prop('disabled', true);
						$('#mds-apply').prop('disabled', true);
						$('#mds-network-info').prop('disabled', true);
						$('#network_option').prop('disabled', true);
					},
					function(jqXHR,textStatus,errorThrown) {
						console.error(jqXHR,textStatus,errorThrown);
					}
				);
		},
		
		
		/**
		 * 이벤트 목록 메서드
		 */
		listener : function() {
			$('#mds-disk-name').off('change').on('change',function(){
				$('#mds-disk-info').text($('option:selected', this).attr('size'));
			});
			
			$('#mds-apply').off('click').on('click',function(){
				MDS_Setting.send_mds_info();
			});
			
			$('#mds-reset').off('click').on('click',function(){
				MDS_Setting.init();
			});
			
			$('#mds-network-info').off('change').on('change',function(){
				//console.table($(this))
				var network_type = $('option:selected', this).data('type');
				var network_name = $(this).val();
				var is_tcp = network_type === 'infiniband' ? 'o2ib1' : 'tcp';
				var result = 'options lnet networks="'+is_tcp+'('+network_name+')"';
				$('#network_option').val(result);
			});
		}
}