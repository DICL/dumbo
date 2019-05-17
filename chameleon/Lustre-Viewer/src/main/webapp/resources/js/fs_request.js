$(function(){
	Fs_Request.init();
});

var Fs_Request = {
		init : function() {
			Fs_Request.listener();
		},
		
		oss_nodes : null,
		
		/**
		 * shkim 20181215
		 * 암바리를 통하여 얻은 정보를 디비에 적재하는 메서드
		 */
		syncLustreTable : function(fs_name,callback) {
			changeLoadingText("Get Ambari Server");
			$.ajax({
				url: contextPath+'/api/v1/ambari/createLustreNodeForAabari',
				type: "GET",
				data: {
					fs_name : fs_name,
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
		 * 암바api을 통하여 mds 정보들을 가져오느 메서드
		 */
		getAmbariForMDSDisks : function() {
			return $.ajax({
				data : {},
				url: contextPath+'/api/v1/lustre/getAmbariForMDSDisks',
				type: "POST",
			})
		},
		
		/**
		 * MGT 디스크정보을 가져오는 매서드 
		 */
		getMGTDisk : function() {
			return $.ajax({
				data : {},
				url: contextPath+'/api/v1/lustre/get_MGT_Disk',
				type: "POST",
			})
		},
		
		/**
		 * lustre file system 내용 갱신
		 */
		updateFilesystem : function(fs_stap,callback) {
			console.log("ajaxData", fs_stap);
			var ajaxData = {
					num : fs_stap['fs_data']['num'],
					fs_step : fs_stap['fs_data']['fs_step'],
			}
			
//			if(typeof fs_stap['fs_name'] !== 'undefined'){
//				ajaxData['fs_name'] = fs_stap['fs_name']
//			}
			
			
			
			return $.ajax({
				data : ajaxData,
				url: contextPath+'/api/v1/lustre/setFileSystem',
				type: "POST",
				success : function(res) {
					if(res.status){
						callback();
					}
				},
			})
		},
		
		
		getFilesystem : function(data, callback) {
			return $.ajax({
				data : data,
				url: contextPath+'/api/v1/lustre/viewFileSystem',
				type: "POST",
				success : function(res) {
					callback(res);
				},
			});
		},
		
		/**
		 * 공통적으로 동작되는 메서드
		 */
		add_file_system : function(fs_stap) {
			var target = '.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"]';
			if(typeof fs_stap === 'undefined' || !fs_stap || fs_stap['step'] == ''){
				var fs_stap = {};
				fs_stap['step'] = 'step0';
			}
			
			// 모달창 그리기
			templete_html($(target)[0],"file_systems/"+fs_stap['step'],fs_stap,function(){
				// 메뉴그리기
				templete_html($('#fs-add-nav')[0],"file_systems/menu",fs_stap,function(){
					
					try {
						$('[data-step="'+fs_stap['step']+'"').addClass('active');
						
						// 로그확인
						console.log('request_'+fs_stap.step+' ==>',fs_stap);
						
						// 이벤트 확인
						Fs_Request.listener(fs_stap);
						
						// 전용 함수 실행
						Fs_Request['request_'+fs_stap.step+'_init'](fs_stap);
						// 로그 보기 버튼 이벤트
						operationsRunning.listener();
						
					 } catch (exception) {
						console.log(exception);
					 }
				});
			});
		},
		
		/**
		 * 에러시 처리되는 메서드
		 */
		errorAjax : function(jqXHR,textStatus,errorThrown) {
			console.error(jqXHR,textStatus,errorThrown);
			hideLoading();
		},
		
		/**
		 * 0 단계 시작시 동작되는 메서드 
		 */
		request_step0_init : function(fs_data) {
			showLoading("get Ambari LustreMDSMgmtService Service .....");
			// Ambari 을 통하여 MDS노드 정보 가져오기
			Fs_Request.getAmbariForMDSDisks()
			.then(
					function(data, textStatus, jqXHR) {
						var mgs_host_name = _.keys( data.result)[0];
						var disk_list = data.result[mgs_host_name];
						
						// 190107 je.kim 디스크 그리기 로직 추가
						$('#mgs-disk-name').empty();
						var html = '';
						html += "<option value=\"\" size=\"\">Select DISK</option>";
						_.each(disk_list,function(disk_info) {
							html += "<option value=\""+disk_info.name+"\" size=\""+disk_info.size+"\" >";
							html += disk_info.name + '('+disk_info.size+')';
							html += "</option>";
						});
						$('#mgs-disk-name').html(html);
						$('.mgs-hostname').text(mgs_host_name);
						
						Fs_Request.getMGTDisk().then(
							function(data, textStatus, jqXHR) {
								if(typeof data.data !== 'undefined'){
									
									if(typeof _.find($('#step0 #mgs-disk-name option'),function(obj){ return $(obj).val() === data.data.disk_name }) === 'undefined' ){
										console.log('add mgt disk')
										$('#step0 #mgs-disk-name').append('<option value="'+ data.data.disk_name +'" size="'+data.data.disk_size+'" >' + data.data.disk_name + ' ('+ data.data.disk_size +')</option>');
									}
									
									console.log('install mgt disk =>' ,data)
									$('#mgs-disk-name').val(data.data.disk_name);
									$('#mgs-disk-name').prop('disabled',true);
								}
								hideLoading();
							},Fs_Request.errorAjax
						);
						
						
						
						// 190124 je.kim filesystem name은 8자로 제한
						$('#step0.modal-dialog #fs_name').off('keyup').on('keyup',function(e){
							var file_system_name = $(this).val();
							
							if(file_system_name.length >= 8){
								console.log(file_system_name)
								$(this).val(file_system_name.substring(0,8));
							}
						});
						
						
						// step0 에서 next 버튼 클릭시
						$('#'+fs_data['step']+' button[type=submit]').off('click').on('click',function(e){
							showLoading('set lustre file system');
							Fs_Request.send_request_step0(fs_data).then(
								function(data, textStatus, jqXHR) {
									hideLoading();
									console.log('request_step0_send_result ==>',data);
									if(data.status){
										
										// je.kim 성공시 가지고 있는 fs_name 을 이용하여 다시 서버로 읽어오고
										// 기존 데이터를 교체함
										// fs_name 은 유니크 키로 교체하고 , 각 스텝별로 성공시 스텝을 올리고, 다시 리로드 처리
										Fs_Request.getFilesystem({fs_name : $('#fs_name').val()},function(get_data){
											console.log(get_data)
											fs_data['fs_data'] = get_data;
											fs_data['step'] = 'step1';
											fs_data['move-step'] = 'step0';  
											fs_data['fs_name'] = $('#fs_name').val();
											Fs_Request.add_file_system(fs_data);
										});
										
										
										
//										if(typeof fs_data['fs_data'] !== 'undefined'){
//											fs_data['fs_data']['fs_name'] = $('#fs_name').val()
//										}else{
//											fs_data['fs_data'] = {
//												'fs_name' : $('#fs_name').val(),
//											}
//										}
									}
								},
								Fs_Request.errorAjax
							);
						});
					},
					Fs_Request.errorAjax
			);
			
			
			
		},
		
		/**
		 * 서버에 fs 이름 보내기
		 */
		send_request_step0 : function(data) {
			if($('#fs_name').val() === ''){
				alert('plase file system name');
				$('#fs_name').focus();
				return false;
			}
			var sendData = {
					// file system information
					fs_name : $('#fs_name').val(),
					fs_step : 1,
					num : (
						typeof data['fs_data'] !== 'undefined'
						&& data['fs_data']['num'] !== 'undefined'
					)? data['fs_data']['num'] : null,
					// mgt information
					disk_name : $('#mgs-disk-name').val(),
					disk_type : 'MGT',
			}
			
			return $.ajax({
				data : sendData,
				url: contextPath+'/api/v1/lustre/add_filesystem_and_add_MGTDisk',
				type: "POST",
			})
		},
		
		/**
		 * 1 단계 시작시 처음으로 동작되는 메서드
		 */
		request_step1_init : function(fs_data) {
			showLoading("check tables");
			$('#mds-disk-info').text('');
			console.log("request_step1_init fs_data",fs_data);
			
			// 190208 je.kim 초기 노드정보를 저장하기 위한 변수, Next 버튼 클릭시 사용됨
			var mds_node_info = null;
			
			// 시작후 MDS 노드 정보 가져오기
			// Fs_Request.get_MDS_nodes(fs_data['fs_data']['num'],function(lustre_nodes) {
			// 네트워크 세팅 , 기존 MDS 디스크 세팅 여부도 확인하기 위하여 전체 MDS 정보들을 가져오기
			Fs_Request.get_MDS_nodes(undefined,function(lustre_nodes) {
				// ajax 가 실행될 횟수 , ajax 가 실행될때 마다 ajax_count 가 1씩 증가함
				var ajax_total_count = 2;
				
				var ajax_count = 0;
				// 181227 je.kim 해당 fs_num 을 기준으로 필터링
				mds_node_info = _.filter(lustre_nodes,function(item){ return item.node_type === "MDS" && item.file_system_num === fs_data.fs_data.num });
				console.log('file system num ->',fs_data['fs_data']['num'],'->',mds_node_info);
				// 만약 가져온 정보가 없을 경우 새로운 겍체로 초기화
				mds_node_info = mds_node_info.length > 0 ? mds_node_info[0] : {};
				Fs_Request.nodes = mds_node_info;
				
				// 상단 타이틀 출력
				$('.mds-hostname').text(mds_node_info['host_name'])
				changeLoadingText("Read MDS Information.....");
				
				// 해당 MDS 노드의 디스크 정보들을 읽어오기
				Fs_Request.get_mds_disk(fs_data['fs_data']['num'],function(disk_list) {
					ajax_count ++; // 성공시 ajax_count 을 1 증가
					// ajax 을 실행후 결과을 셀렉트 박스로 출력
					
					Fs_Request.grid_disk_options(disk_list[mds_node_info.host_name],mds_node_info.host_name,lustre_nodes);
					
					if(mds_node_info['disk_list'].length > 0){
						//$('#mds-disk-name').val(mds_node_info['disk_list'][0]['disk_name']);
						// shkim 20181211
						// 설치 단에서는 MDT 설치 티스크 선택 가능하여야함.
						//$('#mds-disk-name').prop('disabled', true);
						$('#mds-apply').prop('disabled', true);
					}
					
					// ajax_count 가 ajax_total_count 보다 같거나 클경우 로딩화면 종료
					if(ajax_count >= ajax_total_count){
						hideLoading();
						Fs_Request.listener(fs_data);
					}
					
				})
				
				// 해당 MDS 노드의 네트워크 정보 가져오기
				Fs_Request.get_mds_network(fs_data['fs_data']['num'],function(network_list) {
					ajax_count ++; // 성공시 ajax_count 을 1 증가
					//var tmp_network_list = _.map(network_list[mds_node_info.host_name],function(network_name){return network_name.split(':')[0]})
					
					// 전체 MDS 노드를 읽어온 후에 셀렉트 박스 만들기
					// ajax 을 실행후 결과을 셀렉트 박스로 출력
					Fs_Request.grid_network_options(network_list[mds_node_info.host_name]);
					// 네트워크 세팅한 MDS 노드 탐색
					var mds_network_setting_node = _.filter(lustre_nodes,function(mds_column){ return typeof mds_column.network_device !== 'undefined' && mds_column.network_device !== null })
					
					try {
						// 만약  네트워크 세팅한 MDS 노드가 있을경우 네트워크 설정을 못하게 disabled 설정
						if(typeof mds_network_setting_node[0] !== 'undefined'){
							$('#mds-network-info').val(mds_network_setting_node[0].network_device);
							$('#mds-network-info').prop('disabled',true);
						}
					} catch (e) {
						conslole.log(e)
					}
					
					
					// 이전 소스를 복사 & 붙여넣기 한 부분이라 주석
//					if(typeof mds_node_info['network_device'] !== 'undefined' && mds_node_info['network_device'] !== null && mds_node_info['network_device'] != ""){
//						$('#mds-network-info').val(mds_node_info['network_device']);
//						$('#network_option').val(mds_node_info['network_option']);
//						// 수정후 주석 풀
//						//$('#mds-network-info').prop('disabled', true);
//					}
					
					// ajax_count 가 ajax_total_count 보다 같거나 클경우 로딩화면 종료
					if(ajax_count >= ajax_total_count){
						hideLoading();
						Fs_Request.listener(fs_data);
					}
				})
			});
			
			// MDS 세팅에서 NEXT 버튼클릭시
			$('#'+fs_data['step']+' button[type=submit]').off('click').on('click',function(e){
				// 로딩화면 출력
				showLoading('MDS Setting');
				
				// 설정한  MDS 정보 전송
				Fs_Request.send_request_step1(fs_data,mds_node_info).then(
					function(data, textStatus, jqXHR) {
						// 완료 후 로딩화면 숨기기
						hideLoading();
						console.log('request_step1_send_result ==>',data);

						$('#mds-disk-name').prop('disabled', true);
						$('#mds-apply').prop('disabled', true);
						$('#mds-network-info').prop('disabled', true);
						$('#network_option').prop('disabled', true);
						
						// je.kim 성공시 가지고 있는 fs_name 을 이용하여 다시 서버로 읽어오고
						// 기존 데이터를 교체함
						// fs_name 은 유니크 키로 교체하고 , 각 스텝별로 성공시 스텝을 올리고, 다시 리로드 처리
						Fs_Request.getFilesystem({fs_name : fs_data['fs_data']['fs_name']},function(get_data){
							fs_data['fs_data'] = get_data;
							fs_data['step'] = 'step2';
							fs_data['move-step'] = 'step1';  
							console.log("step1 fs_data", fs_data);
							Fs_Request.add_file_system(fs_data);
						});
						
//						if(data.status){
//							fs_data['step'] = 'step2';
//							fs_data['move-step'] = 'step1';
//							fs_data['fs_data']['fs_step'] = 2;
//						
//							if(typeof fs_data['fs_data'] !== 'undefined'){
//								//fs_data['fs_data']['fs_name'] = fs_data['fs_name'];
//								fs_data['fs_data']['fs_step'] = 2
//							} 
//							
//							console.log("step1 fs_data", fs_data);
//							Fs_Request.add_file_system(fs_data);
////							Fs_Request.updateFilesystem(fs_data,function() {
////								Fs_Request.add_file_system(fs_data);
////							});
//						}
					},
					Fs_Request.errorAjax
				);
			});
			
		},
		
		/**
		 * shkim 20181212
		 * 서버에 mds, mdt 정보 보내기
		 */
		send_request_step1 : function(fs_data,lustre_nodedata) {
//			showLoading('set MDS system');
//			var mds_node_info = _.filter(lustre_nodes,function(item){ return item.node_type === "MDS" });
//			mds_node_info = mds_node_info.length > 0 ? mds_node_info[0] : {};
			var tmpdata = lustre_nodedata;
//			tmpdata = _.filter(tmpdata,function(target_node){ 
//				return target_node.node_type === "MDS" &&  target_node.file_system_num == data['fs_data']['fs_num']
//			});
			
			console.log("mds_data", tmpdata);
			//console.log("send_request_step1 arg data", data);
			var send_data = {};
			
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
			
			send_data['host_name'] = tmpdata.host_name;
			send_data['index'] = tmpdata.index;
			send_data['disk_type'] = 'MDT';
			send_data['node_type'] = 'MDS';
			send_data['index'] = 0;
			send_data['disk_name'] = $('#mds-disk-name').val();
			send_data['disk_size'] = $('#mds-disk-name option:selected').attr('size');
			send_data['network_device'] = $('#mds-network-info').val();
			send_data['fs_name'] = fs_data['fs_data']['fs_name'];
			send_data['file_system_num'] = fs_data['fs_data']['num'];
			
			var network_type = $('#mds-network-info option:selected').data('type').split('/')[1];
			var is_tcp = network_type === 'ether' ? 'tcp' : 'o2ib1';
			fs_data['network_option'] = 'options lnet networks="'+is_tcp+'('+$('#mds-network-info').val()+')"';
			console.log("filesystem data =>", fs_data);
			console.log("server sned data =>", send_data);
//			return null;
			
			return $.ajax({
					url: contextPath+'/api/v1/lustre/mdsAddDisk_fs',
					type: "POST",
					data: send_data
			})
		},
		
		
		/**
		 * shkim 20181212
		 * 2 단계 시작시 처음으로 동작되는 메서드
		 */
		request_step2_init : function(fs_data) {
			showLoading("OSS Setting");
//			$('#mds-disk-info').text('');
			console.log("request_step2_init fs_data",fs_data);
			// 디비에서 OSS 정보들을 읽어와서
			
			Fs_Request.get_lustre_oss_nodes(
					//fs_data.fs_data.num, // fs_num 으로 필터링
					// 190131 je.kim undefined -> fs_num
					// 190211 je.kim disk , network 설정값 가져오기 위하여 전체 노드정보 가져오기
					 undefined,
					function(node_list, textStatus, jqXHR) {
						changeLoadingText("Read OSS Information.....");
						Fs_Request.oss_nodes = node_list;
						// 전체 데이터는 백업
						var all_lustre_nodes = node_list;
						// je.kim file system 으로 필터링
						node_list = _.filter(
							node_list , function(item){
								return item.file_system_num === fs_data.fs_data.num;
							}
						);
						
						var result = {};
						
						// 190211 je.kim 기존의 설정된 network 탐색 
						var network_setting_nodes = _.filter(
								all_lustre_nodes , function(item){
									return typeof item.network_device !== 'undefined';
								}
							);
						
						// 190211 je.kim 기존의 설정된 network 가 있을경우 정보 갱신
						// 예외처리 추가
						try {
							node_list = _.map(
									node_list , function(item){
										var matching_oss_node = _.find(network_setting_nodes,function(find_item){
											return find_item.host_name === item.host_name;
										});
										item.network_device = 
											(typeof matching_oss_node !== 'undefined') 
												? matching_oss_node.network_device : undefined;
										return item;
									}
								);
						} catch (e) {
							node_list = node_list;
						}
						
						
						// SSH 을 이용하여 네트워크 정보들을 읽어옴
						Fs_Request.get_oss_network(
							fs_data.fs_data.num, // fs_num 으로 필터링
							function(oss_networks){							
								// 읽어온 네트워크 정보들을 기존노드 정보에다가 갱신
								result = _.map(node_list,function(node_info){
									//node_info.network_list = oss_networks[node_info['host_name']]
									
									var temp_oss_network_list = _.map(oss_networks[node_info['host_name']],function(item){return {name : item.split(':')[0], type : (item.split(':').length > 1) ? item.split(':')[1] : ''} })
									node_info.network_list = temp_oss_network_list
									
									return node_info;
								})
							// 전체 oss 디스크정보들을 읽어와서
							Fs_Request.get_oss_disk(
								//undefined,
								fs_data.fs_data.num, // fs_num 으로 필터링
								function(oss_disks){
									// 전체 디스크 숫자 카운팅
									var disk_total_count = _.reduce(oss_disks, function(memo, num){ return memo + num.length; }, 0);
									Fs_Request['disk_total_count'] = disk_total_count;
									Fs_Request['OST_info']={}
									// 탭 이름 및 임시 변수에 저장
									for (var i = 0; i < disk_total_count; i++) {
										var temp_ost_name = "OST"+(i);
										Fs_Request['OST_info'][temp_ost_name]=null;
									}
									
									var result = {};
									// 읽어온 디스크정보들을 토대로 기존 노드 정보에다가 갱신
									result = _.map(node_list,function(node_info){
										
										// 호스트 명으로 전체 노드 필터
										var filter_disklist_host_for_node = _.filter(all_lustre_nodes,function(all_lustre_node_item){return all_lustre_node_item.host_name === node_info['host_name']});
										// 디비에 저장되어 있던 디스크 정보 추출
										filter_disklist_host_for_node = _.map(filter_disklist_host_for_node,function(all_lustre_node_item){return all_lustre_node_item.disk_list});
										// 전체 배열 통합 & 디스크 이름만 추출
										filter_disklist_host_for_node = _.map(_.flatten(filter_disklist_host_for_node),function(all_lustre_node_item){return all_lustre_node_item.disk_name});
										// console.log('filter_disklist_host_for_node',filter_disklist_host_for_node,all_lustre_nodes)
										
										node_info.disk_info_list = 
											_.map(
												oss_disks[node_info['host_name']],
												function(disk_info){
													// 디비에 저장되어 있었던 디스크명을 조회하여 만약 해당 디스크 명이 있다면 사용중으로 간주하고 disable 처리
													disk_info.is_used =  typeof _.find(filter_disklist_host_for_node,function(find_item){return disk_info.name === find_item}) !== 'undefined';
													return disk_info;
												}
											);
										node_info.disk_total_count = disk_total_count;
										return node_info;
									})
									
									// 최종 완성된 데이터
									console.log("result", result);
									
									// 완성된 데이터를 토대로 설정페이지 업데이트
									Fs_Request.grid_oss_config_page(result,function(){
										
										$('#'+fs_data['step']+' button[type=submit]').off('click').on('click',function(e){
											showLoading('OSS Setting');
											// 190211 je.kim 러스터 노드정보도 추가
											Fs_Request.send_request_step2(fs_data,result)
											.then(
												function ( data, textStatus, jqXHR) {
													console.log("send_oss_disk ==>",data, textStatus, jqXHR)
	//												alert('OSS Setting');
	//												operationsRunning.showViewLog(data[0]['data']);
	//												var target = '.bd-managerque-modal-lg';
	//												$(target).modal('show');
													hideLoading();
													var result = _.reduce(data, function(memo, num){ return memo.status && num.status; });
													
													// je.kim 성공시 가지고 있는 fs_name 을 이용하여 다시 서버로 읽어오고
													// 기존 데이터를 교체함
													// fs_name 은 유니크 키로 교체하고 , 각 스텝별로 성공시 스텝을 올리고, 다시 리로드 처리
													if(result){
														Fs_Request.getFilesystem({fs_name : fs_data['fs_data']['fs_name']},function(get_data){
															fs_data['fs_data'] = get_data;
															fs_data['step'] = 'step3';
															fs_data['move-step'] = 'step2';  
															console.log("step2 fs_data", fs_data);
															Fs_Request.add_file_system(fs_data);
														});
													}
													
													
//													if(result){
//														fs_data['step'] = 'step3';
//														fs_data['move-step'] = 'step2';
//														fs_data['fs_data']['fs_step'] = 3
//														if(typeof fs_data['fs_data'] !== 'undefined'){
//															//fs_data['fs_data']['fs_name'] = fs_data['fs_name'];
//															fs_data['fs_data']['fs_step'] = 3;
//														}
//														
//	//													Fs_Request.updateFilesystem(fs_data,function() {
//	//														Fs_Request.add_file_system(fs_data);
//	//													});
//														
//													}
												},
												Fs_Request.errorAjax
											);
											// 리스너 동작
											
										});
	
										Fs_Request.listener(fs_data);
									})
								}
							)
					});
				}
			)			
		},
		
		
		/**
		 * oss 정보들을 전송하는 메서드
		 */
		send_request_step2 : function(fs_data,lustre_node_list) {
			var send_node = {};
			_.each(Fs_Request.oss_nodes,function(item,select_index){
				send_node[item['host_name']] = {};
				send_node[item['host_name']]['disk_list'] = [];
				send_node[item['host_name']]['host_name'] = item['host_name'];
				send_node[item['host_name']]['network_device'] = $('#'+item['host_name']+'-network').val();
				send_node[item['host_name']]['node_type'] = 'OSS';
				send_node[item['host_name']]['index'] = item['index'];
			})
			
			_.each($('#step2 .ost-name'),function(ost_obj){
				if($(ost_obj).val() !== 'none' && $(ost_obj).prop('disabled') === false){
					var host_name = $(ost_obj).attr('host-name');
					var select_index = $(ost_obj).attr('index');
					var disk_index = $(ost_obj).val();
					var target_disk_select_id = '#OSS-'+host_name+'-'+select_index+'-diskname';
					var disk_name = $('#step2 ' + target_disk_select_id).val();
					console.log(target_disk_select_id)
					
					var tmp = {
							disk_type : 'OST',
							index : disk_index,
							disk_name : disk_name,
							disk_size : $('#step2 ' + target_disk_select_id + ' option:selected').attr('size'),
					}
					send_node[host_name]['disk_list'].push(tmp)
				}
			});
			
			//console.log(fs_data)
			// object -> array 로 전환
			// je.kim 190211 file_system_num 추가 , node num 추가
			var tem_list = _.map(send_node,function(item){
				var temp_lustre_node_info = _.find(
						lustre_node_list,
						function(lustre_node_info){
							return lustre_node_info.host_name === item.host_name;
						}
				);
				// num 설정
				item.num = temp_lustre_node_info.num;
				item.file_system_num = fs_data['fs_data']['num'];
				return item;
			})
			var data = {};
			data['list'] = tem_list;
			data['file_system_num'] = fs_data['fs_data']['num'];
			console.log(fs_data,data)
//			data['fs_name'] = fs_data['fs_data']['fs_name'];
//			data['fs_num'] = fs_data['fs_data']['fs_num'];
			
			
			return $.ajax({
				url: contextPath+'/api/v1/lustre/ossAddDisk_fs',
				type: "POST",
				data: JSON.stringify( data ),
				dataType: "json",
			    contentType : 'application/json'
			})
		},
		
		
		/**
		 * step3 페이지 로딩시 시작되는 페이지
		 */
		request_step3_init : function(fs_data) {
			showLoading('get lustre information')
//			hideLoading();
//			Client_Setting.listener();
			
			// 전체 노드정보 가져오기
			Fs_Request.getClientList(undefined).then(
			//Fs_Request.getClientList(fs_data['fs_data']['num'] ).then(
				function( lustre_cilent_nodes , textStatus, jqXHR) {
					// 해당 파일시스템의 lustre client node 정보 가져오기 
					var filesystem_filter_client_node =
						_.filter(lustre_cilent_nodes,function(node_information){ return node_information.file_system_num === fs_data['fs_data']['num']});
					// 네트워크 설정이 되어 있는 lustre client node 정보 가져오기
					var network_setting_node =
						_.filter(lustre_cilent_nodes,function(node_information){ return typeof node_information.network_device !== 'undefined' });
					
					
					// 네트워크 설정이 이미 되었으면 network_device 설정
					filesystem_filter_client_node = 
						_.map(filesystem_filter_client_node,function(node_information){
							var temp_find_node = _.find(
									network_setting_node,
									function(find_node_info){
										return node_information.host_name === find_node_info.host_name;
									}
							);
							node_information.network_device = (typeof temp_find_node !== 'undefined') ? temp_find_node.network_device : undefined;
							return node_information;
						});
					
					
					Fs_Request.gridClient(fs_data , filesystem_filter_client_node, function(){
						
						hideLoading();
						
						// step 3 에서 next 버튼 클릭시 동작되는 이벤트
						$('#'+fs_data['step']+' button[type=submit]').off('click').on('click',function(e){
							showLoading('Client Setting');
							
							Fs_Request.send_request_step3(fs_data,filesystem_filter_client_node).then(
								function ( data, textStatus, jqXHR) {
									hideLoading();
									console.log("send_client_info ==>",data, textStatus, jqXHR)
									//var result = _.reduce(data, function(memo, num){ return memo.status && num.status; });

									Fs_Request.getFilesystem({fs_name : fs_data['fs_data']['fs_name']},function(get_data){
										fs_data['fs_data'] = get_data;
										fs_data['fs_data']['fs_step'] = 4
										fs_data['step'] = 'step4';
										fs_data['move-step'] = 'step3';  
										console.log("step3 fs_data", fs_data);
										Fs_Request.add_file_system(fs_data);
									});
									
//									if(result){
//										fs_data['step'] = 'step4';
//										fs_data['move-step'] = 'step3';
//										fs_data['fs_data']['fs_step'] = 4
//										if(typeof fs_data['fs_data'] !== 'undefined'){
//											//fs_data['fs_data']['fs_name'] = fs_data['fs_name'];
//											fs_data['fs_data']['fs_step'] = 4
//										}
//										
////										Fs_Request.updateFilesystem(fs_data,function() {
////											Fs_Request.add_file_system(fs_data);
////										});
//										
//									}
								},
								Fs_Request.errorAjax
							);
						});
						Fs_Request.listener(fs_data);
						
					});
				}
				,Fs_Request.errorAjax
			);
		},
		
		/**
		 * cilent 의 정보들을 서버에 전송하는 메서드
		 */
		send_request_step3 : function(fs_data,lustre_nodes) {
			//console.log(fs_data); return;
			var host_list = Fs_Request.host_list;
			if(host_list.lenth <= 0){
				return false;
			}
			for (var i = 0; i < host_list.length; i++) {
				var host_name = host_list[i];
				if(!$('#' +  host_name + '-network').val()){
					alert('Please enter a network name');
					$('#' +  host_name + '-network').focus();
					return false;
				}
				if(!$('#' +  host_name + '-network_option').val()){
					alert('Please enter a config');
					$('#' +  host_name + '-network_option').focus();
					return false;
				}
			}
			var send_data = {};
			
			send_data['file_system_num'] = fs_data['fs_data']['num'];
			send_data['list'] = _.map(host_list,function(host_name){
				
				var lustre_node_info = _.find( lustre_nodes ,function(tmp_node_infomation){
					return tmp_node_infomation.host_name === host_name;
				});
				return {
					num : lustre_node_info.num,
					host_name : host_name,
					network_device : $('#' +  host_name + '-network').val(),
					node_type : 'CLIENT',
					network_option : $('#' +  host_name + '-network_option').val(),
					lustre_client_folder :  $('#client-mount-name').val(),
					file_system_num : fs_data['fs_data']['num'],
				}
			});
			console.log('client send data ->',send_data)
			
			
			
			return $.ajax({
				url: contextPath+'/api/v1/lustre/clientSetting_fs',
				type: "POST",
				data: JSON.stringify( send_data ),
				dataType: "json",
			    contentType : 'application/json',
			})
		},
		
		
		/**
		 * 
		 */
		request_step4_init : function(fs_data) {
			console.log('request_step4_init',fs_data)
			Fs_Request.grid_summary_node(
					fs_data['fs_data']['num']
					,$('#step4 #fs-add-container')[0]
					,function(){
						$('#'+fs_data['step']+' button[type=submit]').off('click').on('click',function(e){
							fs_data['step'] = 'step5';
							fs_data['move-step'] = 'step4';
							fs_data['fs_data']['fs_step'] = 5
							if(typeof fs_data['fs_data'] !== 'undefined'){
								fs_data['fs_data']['fs_name'] = fs_data['fs_name'];
								fs_data['fs_data']['fs_step'] = 5
							}
							Fs_Request.updateFilesystem(fs_data,function() {
								//Fs_Request.add_file_system(fs_data);
								IndexPage.init();
								$('.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"]').modal('toggle')
							});
						});
					}
			);
		},
		
		
		/**
		 * 최종결과 출력
		 * file_system_num : file system number
		 * gird_target : 화면에 그릴 DOM object
		 * callback : 콜백함수
		 */
		grid_summary_node : function(file_system_num,gird_target,callback) {
			showLoading('Read Lustre Nodes');
			Fs_Request.getFilesystem({num : file_system_num},function(file_system_data){
				console.log('grid_summary_node',file_system_data)
				Fs_Request.getLustreNodeList(file_system_data.num)
				.then(
					function ( lustre_node_list, textStatus, jqXHR) {
						//var grid_data = _.groupBy(lustre_node_list,function(item){return item.node_type;});
						//lustre_node_list = _.filter(lustre_node_list, function(item){ return item.file_system_num === file_system_num });
						console.log(lustre_node_list);
						var grid_data = {
								lustre_node_list : lustre_node_list,
								file_system : file_system_data
						}
						console.log(grid_data);
						templete_html(gird_target,"file_systems/step4-summary",grid_data,function(){
							hideLoading();
						});
						
						callback();
					}
					,Fs_Request.errorAjax
				);
			})
			
			
		},
		
		
		/**
		 * step4 시작시 전체 lustre node 정보 가져오기
		 */
		getLustreNodeList : function(file_system_num) {
			return $.ajax({
				url: contextPath+'/api/v1/ambari/getLustreNodes',
				type: "POST",
				data: {
					'file_system_num': file_system_num,
				}
			})
		},
		
		
		
		/**
		 * CLIENT 노드들의 네트워크 정보들을 읽어오는 메서드
		 */
		getClientNetworkList : function(fs_data,callback) {
			changeLoadingText("read lustre networks");
			$.ajax({
				url: contextPath+'/api/v1/lustre/getNetWorkDevice',
				type: "POST",
				data: {
					'node_type' : 'CLIENT',
					'file_system_num' : fs_data['fs_data']['num']
				}
			}).then(function (data, textStatus, jqXHR) {
				console.log("get_oss_network ==>",data, textStatus, jqXHR)
				callback(data);
			},function(jqXHR,textStatus,errorThrown) {
				console.error(jqXHR,textStatus,errorThrown);
			})
		},
		
		/**
		 * client node 정보 가져오기
		 */
		getClientList : function(file_system_num) {
			return $.ajax({
				url: contextPath+'/api/v1/ambari/getLustreNodes',
				type: "POST",
				data: {
					node_type : 'CLIENT',
					'file_system_num': typeof file_system_num !== 'undefined' ? file_system_num : null
				}
			})
		},
		
		
		/**
		 * 가져온 client 정보를 토대로 화면에 그리는 메서드
		 */
		gridClient : function(fs_data ,data , callback) {
			console.log('gridClient ==>',data)
			// 호스트 리스트 들을 변수에 저장
			Fs_Request.host_list = _.map(data,function(item){return item.host_name});
			// 화면에 그릴 dom 겍체
			target = document.getElementById("client_page")
			var node_list = data;
			
			Fs_Request.getClientNetworkList( fs_data,function(network_list) {
				var result = _.map(node_list,function(node_info){
					var temp_oss_network_list = _.map(network_list[node_info['host_name']],function(item){return {name : item.split(':')[0], type : (item.split(':').length > 1) ? item.split(':')[1] : ''} })
					node_info.network_list = temp_oss_network_list
							
					return node_info;
				})
				templete_html(target,"file_systems/step3-client",result,function(){
					
					// 190311 je.kim 파일시스템 이름으로 변경
					$('#step3 #client-mount-name').val('/' + fs_data['fs_data']['fs_name']);
					
					
					_.each(node_list,function(node_info){
						
						if(typeof node_info.network_device !== 'undefined'){
							
							var network_device_info = 
								_.find(node_info.network_list,function(item){ return item.name === node_info['network_device'] });
							var is_tcp = network_device_info.type === 'infiniband' ? 'o2ib1' : 'tcp';
							var network_setting_name = 'options lnet networks="'+is_tcp+'('+network_device_info.name+')"';
							$('#step3 #'+node_info['host_name']+'-network_option').val(network_setting_name);
							
							
							$('#step3 #'+node_info.host_name+'-network').val(node_info.network_device);
							$('#step3 #'+node_info.host_name+'-network_option').val();
							
							$('#step3 #'+node_info.host_name+'-network').prop('disabled',true);
							$('#step3 #'+node_info.host_name+'-network_option').prop('disabled',true);
							
						}
					});
					callback();
				});
			})
			
			
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
		grid_disk_options : function(data_list,host_name,luster_node_list) {
			var html = "";
			// 해당 호스트만 필터링
			var filter_host_lustre = _.filter(luster_node_list,function(item){ return item.host_name === host_name });
			var host_installed_disk_list = 
				_.flatten(
					_.map(
							filter_host_lustre 
							,function(item){
								return item.disk_list
							}
					)		
				)
			
			html += "<option value=\"\" size=\"\">Select DISK</option>";
			_.each(data_list,function(item){
				// 181227 je.kim 
				// 디비에 있는 노드정보안에 설치되어있는 디스크정보들을 읽어와서 
				// 해당호스트내의 디스크 리스트와 대조하여 일치하면 비활성화 처리
				// 180207 je.kim 하는김에 디스크 이름도 출력
				var matching_host_information =  _.find(
						host_installed_disk_list
						,function(find_item){
							return find_item.disk_name === item.name;
						}
					);
				var is_disabled = 
					typeof matching_host_information !== 'undefined';
				html += "<option value=\""+item.name+"\" size=\""+item.size+"\" "+(is_disabled ? "disabled":"")+" >";
				html += item.name +(is_disabled ? " - " +matching_host_information.disk_type:"")+'('+item.size+')';
				html += "</option>";
			})
			$("#mds-disk-name").html(html);
		},
		
		
		/**
		 * 러스터 노드정보들을 디비에서 가져오기
		 */
		get_MDS_nodes : function(file_system_num,callback) {
			$.ajax({
				url: contextPath+'/api/v1/ambari/getLustreNodes',
				type: "POST",
				data: {
					'file_system_num': (typeof file_system_num !== 'undefined' ) ? file_system_num : null,
					'node_type': 'MDS',
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
		 * MDS 내의 디스크 정보를 읽어오는 메서드
		 */
		get_mds_disk : function(file_system_num,callback) {
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/new_getDiskDevice',
				type: "POST",
				data: {
					'node_type' : 'MDS',
					'file_system_num' : file_system_num,
					'index' : 0,
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
		 * MDS 내의 네트워크 정보를 읽어오는 메서드
		 */
		get_mds_network : function(file_system_num,callback) {
			
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
		 * shkim 20181212
		 * OSS 내의 디스크 정보를 읽어오는 메서드
		 */
		get_oss_disk : function(file_system_num,callback) {
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/getDiskDevice',
				type: "POST",
				data: {
					'node_type' : 'OSS',
					'file_system_num': (typeof file_system_num !== 'undefined') ? file_system_num : null,
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
		
		
		new_get_oss_disk : function(callback) {
			$.ajax({
				url: contextPath+'/api/v1/lustre/new_getDiskDevice',
				type: "POST",
				data: {
					'node_type' : 'OSS',
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
		 * shkim 20181212
		 * OSS Node 디비에서 가져오기
		 */
		get_lustre_oss_nodes : function(file_system_num,callback) {
			$.ajax({
				url: contextPath+'/api/v1/ambari/getLustreNodes',
				type: "POST",
				data: {
					'node_type' : 'OSS',
					'file_system_num' : (typeof file_system_num !== 'undefined') ? file_system_num : null,
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
		 * shkim20181212
		 * OSS 내의 네트워크 정보를 읽어오는 메서드
		 */
		get_oss_network : function(file_system_num,callback) {
			
			$.ajax({
				url: contextPath+'/api/v1/lustre/getNetWorkDevice',
				type: "POST",
				data: {
					'node_type' : 'OSS',
					'file_system_num' : (typeof file_system_num !== 'undefined') ? file_system_num : null,
				}
			}).then(function (data, textStatus, jqXHR) {
				console.log("get_oss_networks ==>",data, textStatus, jqXHR)
				callback(data);
			},function(jqXHR,textStatus,errorThrown) {
				console.error(jqXHR,textStatus,errorThrown);
			})
			
			
		},
		
		/**
		 * shki 20181212
		 * OSS 템블릿 정보를 읽어와서 화면에 그리는 메서드
		 */
		grid_oss_config_page : function(data,callback) {
			showLoading('grid html...')
			console.log("grid_oss_nodes ==>",data)
			
			target = document.getElementById("oss_page_add_filesystem")
			templete_html(target,"file_systems/step2-oss",data,function(){
				
				$('#myTab_add_filesystem li:first-child a').tab('show')
				
				Fs_Request.new_get_oss_disk( function(new_oss_disks){
					_.each(new_oss_disks,function(disklist,hostname){
						var tmphtml = '';
						_.each(disklist,function(diskinfo){
							tmphtml += '<option value="'+ diskinfo.name +'" size="' +diskinfo.size+ '">';
							tmphtml += diskinfo.name;
							tmphtml += '</option>';
						});
						var temp_text = '';
						if(disklist.length > 0){
							temp_text = disklist[0].size;
						}
						_.each($('#step2 .disk-name[host-name='+hostname+']'),function(tem_obj,idx){
							$(tem_obj).html(tmphtml);
							$('#OSS-' + hostname + '-'+ idx +'-size').text(temp_text);
						})
						
					});
				
					_.each(data,function(node_info){
						var i = 0
						
						// 디비에 들어있는 디스크 정보들을 돌면서 변경
						_.each(node_info['disk_list'],function(disk_info){
							console.log("node_info"+i,node_info);
							//  OST 번호 수정
							$('#step2 #OSS-'+node_info['host_name']+'-'+i+'-ost').val(disk_info['index']);
							// 수정못하게 disabled
							$('#step2 #OSS-'+node_info['host_name']+'-'+i+'-ost').prop('disabled',false);
							// 디스크 디바이스명 수정
							$('#step2 #OSS-'+node_info['host_name']+'-'+i+'-diskname').val(disk_info['disk_name']);
							// 수정못하게 disabled
							$('#step2 #OSS-'+node_info['host_name']+'-'+i+'-diskname').prop('disabled',false);
							
							// 사이즈 표시
							$('#step2 #OSS-'+node_info['host_name']+'-'+i+'-size').text( $('#OSS-'+node_info['host_name']+'-'+i+'-diskname option:selected').attr('size') );
							// OST정보들을 임시 변수에 저장
							Fs_Request.OST_info['OST'+disk_info['index']] = disk_info['disk_name'];
							
							i ++;
						})
						
						// 네트워크 디바이스가 설정되어 있을경우 적용및 수정 못하게 처리
						if(node_info['network_device']){
							$('#step2 #'+node_info['host_name']+'-network').val(node_info['network_device']);
							try {
								var network_device_info = 
									_.find(node_info.network_list,function(item){ return item.name === node_info['network_device'] });
								var is_tcp = network_device_info.type === 'infiniband' ? 'o2ib1' : 'tcp';
								var network_setting_name = 'options lnet networks="'+is_tcp+'('+network_device_info.name+')"';
								$('#step2 #'+node_info['host_name']+'-network_option').val(network_setting_name);
							} catch (e) {
								console.log(e)
							}
							$('#step2 #'+node_info['host_name']+'-network').prop('disabled',true);
							$('#step2 #'+node_info['host_name']+'-network_option').prop('disabled',true);
						}
						
					})
				});
				
				

				
				
//				$('[data-toggle=toggle]').bootstrapSwitch({
//					onText : 'Activate',
//					offText : 'Deactivate',
//				});
				
				// 190209 je.kim 충돌방지
				_.each($('#step2 .ost-name'),function(item){
					_.each($(item).children(),function(option){
						if($(option).val() !== 'none'){
							var ost_name = 'OST' + $(option).val();
							if(Fs_Request.OST_info[ost_name] != null && $('option:selected', item).attr('ost') != ost_name){
								$(option).prop( "disabled", true );
							}else{
								$(option).prop( "disabled", false );
							}
						}
					})
				});
				if(typeof callback !== 'undefined'){
					callback();
				}
				hideLoading();
			});
		},
		
		/**
		 * shkim 20181213
		 * OST 셀렉트 를 변경시 동작되는 메서드
		 */
		set_ost_item: function() {
			var tar_get = 'ost-name';
			_.each(Fs_Request.OST_info,function(item,key){
				Fs_Request.OST_info[key] = null;
			})
			
			_.each($('#step2  .'+tar_get),function(item){
				
				var value = $(item).val();
				var host_name = $(item).attr('host-name');
				var index = $(item).attr('index');
				//console.log(item)
				var disk_name = $('#step2 #OSS-'+host_name+'-'+index+'-diskname').val();
				
				if(value !== 'none'){
					Fs_Request.OST_info['OST'+value] = disk_name;
				}
			});
			console.log(Fs_Request.OST_info)
			
			_.each($('#step2 .'+tar_get),function(item){
				_.each($(item).children(),function(option){
					if($(option).val() !== 'none'){
						var ost_name = 'OST' + $(option).val();
						if(Fs_Request.OST_info[ost_name] != null && $('option:selected', item).attr('ost') != ost_name){
							$(option).prop( "disabled", true );
						}else{
							$(option).prop( "disabled", false );
						}
					}
				})
			});
		},
		
		
		/**
		 * 이벤트 확인
		 */
		listener : function(fs_data) {
			console.log("listener fs_data", fs_data);
//			if(typeof fs_data === 'undefined' || fs_data === null){
//				console.warn('fs_data undefined shutdown listener');
//				return false;
//			}

			
			// je.kim 메인 화면에서 file_system list 목록을 클릭시 동작되는 메서드
			$('.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"] button.move-step').off('click').on('click',function(e){
				var target = '.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"]';
				var target_step = $(this).data('move-step');
				var step = (parseInt(target_step.replace('step','')) - 1);
				fs_data['step'] = 'step'+ parseInt(target_step.replace('step',''))
				fs_data['move-step'] = 'step'+ (step < 0)? 0 : step
				//$(target).modal('show');
				Fs_Request.add_file_system(fs_data);
			});
			
//			$('.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"] button.move-step').off('click').on('click',function(e){
//				var target = '.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"]';
//				
//			});
			
			// add file system 버튼을 클릭시 이벤트
			$('#fs-add-btn').off('click').on('click',function(e){
				var target = '.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"]';
				$(target).modal('show');
				
				$('#fs-view #fs-menu').empty();
				$('#fs-view #fs-view-container').empty();
				$('#fs-view #fs-menu').hide();
				$('#fs-view #fs-view-container').hide();
				
				Fs_Request.add_file_system();
			});
			
			// step 4 에서 클라이언트 정보를 선택시 이벤트
			$('#step3 .client-network').off('change').on('change',function(e){
				var host_name = $(this).data('host-name');
				var network_type = $('option:selected', this).data('type');
				var network_name = $(this).val();
				var is_tcp = network_type === 'infiniband' ? 'o2ib1' : 'tcp';
				var result = 'options lnet networks="'+is_tcp+'('+network_name+')"';
				$('#'+host_name+'-network_option').val(result);
			});
			
			// step 2 에서 ost 디스크 를 변경시 동작되는 이벤트 처리
			$('#step2 .disk-name').off('change').on('change',function(){
				var index = $(this).attr('index')
				var host_name = $(this).attr('host-name')
				
				$('#step2 #OSS-'+host_name+'-'+index+'-size').text($('option:selected', this).attr('size'));
			});
			
			$('#step2 .ost-name').off('change').on('change',function(){
				var index = $(this).attr('index')
				var host_name = $(this).attr('host-name')
				
				console.log('OSS-'+host_name+'-'+index+'-diskname')
				
				if($(this).val() === 'none'){
					$('#step2 #OSS-'+host_name+'-'+index+'-diskname').prop( "disabled", true );
				}else{
					$('#step2 #OSS-'+host_name+'-'+index+'-diskname').prop( "disabled", false );
				}
				
				Fs_Request.set_ost_item();
				
				
			});
			
			// step 1 에서 mds 네트워크 정보를 수정시 동작되는 이벤트
			$('#step1 .mds-network-info').off('change').on('change',function(e){
				var host_name = $(this).data('host-name');
				var network_type = $('option:selected', this).data('type');
				var network_name = $(this).val();
				var is_tcp = network_type === 'infiniband' ? 'o2ib1' : 'tcp';
				var result = 'options lnet networks="'+is_tcp+'('+network_name+')"';
				$('#'+host_name+'-network_option').val(result);
			});
			
			// step 3 에서 oss 네트워크 정보 수정시 동작되는 이벤트
			$('#step2 .ost-network').off('change').on('change',function(e){
				var host_name = $(this).data('host-name');
				var network_type = $('option:selected', this).data('type');
				var network_name = $(this).val();
				var is_tcp = network_type === 'infiniband' ? 'o2ib1' : 'tcp';
				var result = 'options lnet networks="'+is_tcp+'('+network_name+')"';
				$('#'+host_name+'-network_option').val(result);
			});
			
			
			$('.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"]').off('hide.bs.modal').on('hide.bs.modal', function (event) {
				 try {
					 IndexPage.init();
				} catch (e) {
					// TODO: handle exception
					console.log(e)
				}
			})
		},
}