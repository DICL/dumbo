$(function(){
	IndexPage.init();
});

var IndexPage = {
		completed_steps : 5,
		node_list : {},
		
		/**
		 * IndexPage.js 시작함수
		 */
		init : function() {
			showLoading('get lustre information')
			IndexPage.check_created_tables(function() {
				IndexPage.get_file_system_list().then(
						IndexPage.grid_fs_list,
						IndexPage.errorAjax
				);
//				IndexPage.syncLustreTable(function() {
//					IndexPage.get_file_system_list().then(
//							IndexPage.grid_fs_list,
//							IndexPage.errorAjax
//					);
//					//IndexPage.listener();
//				})
			})
		},
		
		/**
		 * shkim 20181215
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
		 * shkim 20181215
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
		 * je.kim
		 * 파일시스템 리스트를 가져오는 함수
		 */
		get_file_system_list : function() {
			return $.ajax({
				data : {},
				url: contextPath+'/api/v1/lustre/getFsList',
				type: "POST",
			});
		},
		
		/**
		 * je.kim
		 * 파일시스템 정보들을 읽어와서 화면에 그리는 메서드
		 */
		grid_fs_list : function(data, textStatus, jqXHR) {
			templete_html($('#fs-list')[0],"index_fs_list",data,function(data){
				$('#fs-view #fs-menu').empty().hide();
				$('#fs-view #fs-view-container').empty().hide();
				
				$('[data-fstoggle=toggle]').bootstrapToggle({
					on : 'mount',
					off : 'umount',
					size : 'small',
				});
				
				hideLoading();
				IndexPage.listener();
			});
		},
		
		/**
		 * je.kim
		 * 파일시스템 상세정보를 읽어오는 함수
		 */
		viewFileSystem : function(fs_id) {
			return $.ajax({
				data : {
					num : fs_id,
				},
				type : 'POST',
				url : contextPath+'/api/v1/lustre/viewFileSystem',
			});
		},
		
		
		/**
		 * je.kim
		 * filesystem id 를 이용하여 저장한 노드정보들을 가져오는 함수
		 */
		getFilesystemForNodes : function(fs_id) {
			return $.ajax({
				data : {
					file_system_num : fs_id,
				},
				type : 'POST',
				url : contextPath+'/api/v1/ambari/getLustreNodes',
			});
		},
		
		/**
		 * je.kim
		 * 우측에 화면을 그리는 함수
		 */
		gridNodePage : function(data) {
			console.log(data);
			IndexPage.node_list = _.groupBy(data,function(item){return item.node_type;});
			hideLoading();
			templete_html($('#fs-view #fs-menu')[0],"file_system_views/menu",data,function(data){
				$('#fs-view #fs-menu').show();
				$('#fs-view #fs-view-container').show();
				IndexPage.gridViewPage('MDS_Setting')
				IndexPage.listener();
			});
		},
		
		gridViewPage : function(page_name) {
			console.log("view page name ==>",page_name);
			templete_html($('#fs-view #fs-view-container')[0],"file_system_views/"+page_name,{},function(data){
				$.loadScript('page/'+page_name,function(data){
					console.log('page/'+page_name+'.js import');
				});
				IndexPage.listener();
			});
		},
		
		
		/**
		 * je.kim
		 * 에러시 처리되는 메서드
		 */
		errorAjax : function(jqXHR,textStatus,errorThrown) {
			console.error(jqXHR,textStatus,errorThrown);
		},
		
		
		/**
		 * lustere 마운트 혹은 언마운트 하는 메서드
		 * file_system_num : lustre file system number
		 * is_mount : true (mount) , false (umount)
		 */
		lustre_client_mount_or_umount_folder : function(is_mount, file_system_num) {
			var send_url = (is_mount === true) 
					? contextPath+'/api/v1/lustre/mountClientFolder'  // mount url
					: contextPath+'/api/v1/lustre/umountClientFolder'; // umount url
			return $.ajax({
				data : {
					num : file_system_num,
				},
				type : 'POST',
				url : send_url,
			});
		},
		
		
		/**
		 * lustere 제거하는 메서드
		 */
		remove_lustre_filesystem : function(lustre_file_system_num) {
			var send_url = contextPath+'/api/v1/lustre/removeLustreFilesystem';
			
			return $.ajax({
				data : {
					num : lustre_file_system_num,
				},
				type : 'POST',
				url : send_url,
			});
		}, 
		
		/**
		 * je.kim
		 * 이벤트 처리 함수
		 */
		listener : function() {
			$('#fs-list a.list-group-item:not(.disabled) span.fs-menu-list-name').off('click').on('click',function(e){
				$('#fs-list a.list-group-item').removeClass('active');
				$(this).parent().addClass('active');
				
				$('#fs_num').val($(this).parent().data('num'));
				showLoading('read file system');
				var fs_num = $(this).parent().data('num');
				
				IndexPage.viewFileSystem(fs_num).then(
					function(data, textStatus, jqXHR) {
						
						console.log('viewFileSystem result==>',data)
						// 만약 file system 이 완료가 안되어 있으면 add file system 창 띄우기
						if(data.fs_step < IndexPage.completed_steps){
							hideLoading();
							var set_data = {
								step : 'step' + data.fs_step,
								'move-step' : 'step' + (data.fs_step -1),
								fs_data : data,
							}
							Fs_Request.add_file_system(set_data);
							var target = '.bd-file_system_step-modal-lg[aria-labelledby="file_system_step"]';
							$(target).modal('show');
							$('#fs-list a.list-group-item').removeClass('active');
							
							$('#fs-view #fs-menu').empty();
							$('#fs-view #fs-view-container').empty();
							$('#fs-view #fs-menu').hide();
							$('#fs-view #fs-view-container').hide();
						
						// 완료되었으면 해당 내용 출력
						}else{
							// 노드정보들을 읽어와서
							IndexPage.getFilesystemForNodes(data.num).then(
									IndexPage.gridNodePage, // 화면에 출력
									IndexPage.errorAjax // 에러시 에러처리
							);
						}
					},
					IndexPage.errorAjax
				);
			});
			
			/**
			 * del 버튼 클릭시 이벤트
			 */
			$('#lustre-setting-container #fs-del-btn').off('click').on('click',function(e){
				// 러스트 파일시스템 
				var lustre_file_system_num = $('#fs_num').val();
				if(lustre_file_system_num === ''){
					alert('Select lustre file system');
					return false;
				}
				
				console.log('click #fs-del-btn button',lustre_file_system_num)
				
				// 질의문
				var message = "Are you sure you want to remove lustre file system ??";
				// 만약 승락시
				if(confirm(message)){
					showLoading('Remove Lustre file system');
					IndexPage.remove_lustre_filesystem(lustre_file_system_num).then(function() {
						alert('Remove Lustre Filesystem');
						hideLoading() // 로딩화면 숨기기
						IndexPage.init(); // 리로딩
					});
				}
			});
			
			/**
			 * 마운트 스위치 변경시 이벤트
			 */
			$('#fs-list .set-mount-toggle').off('change').on('change',function(event){
				// 체크여부
				var is_check = $(this).prop('checked');
				// 현재 마운트 상태
				var is_client_mount = $(this).data('is_client_mount');
				// 러스트 파일시스템 
				var lustre_file_system_num = $(this).data('num');
				
				
				console.log('click .set-mount-toggle switch',is_check,is_client_mount)
				
				// 마운트 폴더
				var lustre_client_folder = $(this).data('lustre_client_folder');
				
				// 마운트 질의문
				var mount_message = "Are you sure you want to mount this "+lustre_client_folder+"?";
				// 유마운트 질의문
				var umount_message = "Are you sure you want to umount this "+lustre_client_folder+"?";
				// 질의문
				var message = (is_check) ? mount_message : umount_message;
				// 성공 메세지
				var complete_message = (!is_client_mount)? "mount success" : "umount success";
				
				// 상태 변경시 이벤트
				if(is_check !== is_client_mount){
					
					// 만약 거절시
					if(!confirm(message)){
						// bootstrap toggle 상태를 원상태로 복귀
						if(is_check){
							$(this).bootstrapToggle('off');
						}else{
							$(this).bootstrapToggle('on');
						}
					// 만약 동의시
					}else{
						
						// 기존 상태를 바꿔서
						var switch_mount = !is_client_mount;
						
						// 로딩화면 출력
						showLoading(
								switch_mount ? 
										"mount " + lustre_client_folder + "...."
										:"umount " + lustre_client_folder + "...."
						);
						
						
						// 서버에 전송
						IndexPage
						.lustre_client_mount_or_umount_folder(switch_mount,lustre_file_system_num)
						.then(function() {
							hideLoading() // 로딩화면 숨기기
							// 완료시 성공메세지 출력과
							alert(complete_message);
							// 해당 상태를 변경
							$('#fs-list .set-mount-toggle[data-num="'+lustre_file_system_num+'"]').data('is_client_mount',switch_mount);
						});
					}
				}
			});
			
			
			
			/**
			 * 
			 */
			$('#fs-menu a.nav-link').off('click').on('click',function(e){
				$('#fs-menu a.nav-link').removeClass('active');
				$(this).addClass('active');
				var page_name = $(this).attr('id');
				IndexPage.gridViewPage(page_name);
			});
		},
};