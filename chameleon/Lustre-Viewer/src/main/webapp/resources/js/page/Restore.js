$(function(){
	Restore.init();
});

var Restore = {
		init : function() {
			showLoading('get lustre information')
			Restore.findBackupfiles().then(
					Restore.gridNodes,
					Restore.errorAjax
			);
		},
		
		
		/**
		 * 백업 파일을 찾는 메서드 
		 */
		findBackupfiles : function(file_location) {
			changeLoadingText('Finding Backup Files')
			return $.ajax({
				url: contextPath+'/api/v1/lustre/findBackupfiles',
				type: "POST",
				data: {
					file_location : (file_location) ? file_location : null,
					file_system_num : $('#fs_num').val() ? $('#fs_num').val() : null,
				}
			})
		},
		
		/**
		 * 에러시 처리되는 메서드
		 */
		errorAjax : function(jqXHR,textStatus,errorThrown) {
			console.error(jqXHR,textStatus,errorThrown);
		},
		
		/**
		 * 화면을 그리는 메서드
		 */
		gridNodes : function( data, textStatus, jqXHR) {
			// lustre node 정보들을 이용하여 읽어온 백업파일 정보와 매칭시킴
			var result = [];
			console.log("in1")
			
			// 읽어온 백업파일 정보들을 for-each 문으로 탐색
			_.each(data,function(node_info){
				// lustre node 정보들을 for-each 문으로 탐색
				_.each(node_info.backup_list,function(backup_file_name){
					// 슬러스 문자를 기준으로 백업 파일을 나누기  /tmp/oss0-ost1-20190125.bak.tgz
					var backup_file_name_path_list = backup_file_name.split('/');
					// /tmp 다음의 문자열만 추출  oss0-ost1-20190125.bak.tgz
					var origin_file_name = backup_file_name_path_list[backup_file_name_path_list.length - 1];
					// oss0-ost1-20190125.bak.tgz '-' 문자을 나누어서 ost 문자만 추출하고 대문자로 변경 (mdt도 백업할 가능성이 있어서 OST로 정의안함)
					var disk_type = origin_file_name.split('-')[1].substr(0,3).toUpperCase();
					// oss0-ost1-20190125.bak.tgz '-' 문자을 나누어서 1 문자만 추출
					var disk_index = parseInt(origin_file_name.split('-')[1].substr(3,1));
					// 추출한 OST 문자열과 숫자 1을 참고하여 탐색
					var origin_disk_info = 
						_.find(
							node_info.node_info.disk_list,
							function(item){
								return item.disk_type === disk_type && item.index === disk_index;
							}
						);
					// 읽어온 디바이스 정보와 설치된 디스크 정보들을 대조하여 일치하면 비활화처리
					// 190410
//					node_info.device_list = 
//						_.map(
//							node_info.device_list,
//							function(disk_info) {
//								var temp = disk_info;
//								// 가져온 해당 서버의 디스크 리스트와 디비에 저장되어 있는 디스크 리스트와 대조하여
//								// 일치하는 내용이 없다면(undefined) false, 있으면 true 처리
//								temp.is_disabled = 
//									typeof (
//										_.find(
//											node_info.node_info.disk_list,
//											function(item){
//												return item.disk_name === disk_info.name;
//											}
//										)
//									) !== 'undefined';
//								return temp;
//							}
//						)
					
					//console.log("origin_disk_info",origin_disk_info)
					//console.log("node_info",node_info)
					
					// 사전에 정의하 result 변수에 집어놓을 임시변수 생성 
					var temp_backup_info = {
							name : origin_file_name,
							fullpath_name: backup_file_name,
							node_info : node_info.node_info,
							//origin_disk_name : origin_disk_info.disk_name,
							disk_type : disk_type,
							disk_index : disk_index,
							device_list : node_info.device_list,
							disk_type_name : origin_file_name.split('-')[1],
							node_type_name : origin_file_name.split('-')[0],
					}
					console.log("temp_backup_info",temp_backup_info)
					// 만든 임시 변수에 result 변수에 삽입
					result.push(temp_backup_info);
				});
			});
			
			console.log(data,result)
			
			// 만든 result 변수을 이용하여 화면에 출력
			var target = document.getElementById("Restore-page")
			templete_html(target,"Restore",result,function(){
				hideLoading(); // 다 그리면 로딩 끄기
				Restore.listener(); // 이벤트 감시
			});
			
		},
		
		/**
		 * 서버에 restore 명령보내는 메서드
		 */
		restoreDisk : function() {
			// 셀렉트 박스에 체크가 안되어 있으면 경고창 실행후에 종료
			if($('.backup-files:checked').length === 0){
				console.log('not select radio box');
				return false;
			}
			
			// 체크한 체크박스들을 탐색하면서 셀렉트박스들을 선택했는지 검사
			for (var i = 0; i < $('.backup-files:checked').length; i++) {
				var object = $('.backup-files:checked')[i];
				// 셀렉트 박스 아이디
				var select_id = $(object).data('index') + '-device';
				// 해당 셀렉트 박스가 선택이 안되어 있으면 경고문 출력후 종료
				if($('#'+select_id).val() === ''){
					alert('pleaze select device');
					$('#'+select_id).focus();
					return false;
				}
			}
			
			
			var message = 'Are you sure restore Disk ?? \n\nWarring : The contents of the disk will be deleted !';
			
			// 서버에 보낼 데이터를 생성
			var send_data = _.map(
				// 체크한 체크박스들을 탐색하면서
				$('.backup-files:checked'),
				function(object,index) {
					// 호스트 네임 추출
					var host_name = $(object).data('host_name');
					// 파일네임 추출
					var restore_file = $(object).val();
					// OST1, OST2 디스크 타입 추출
					var disk_type_name = $(object).data('disk_type_name');
					// OST , MDT 추출
					var disk_type = disk_type_name.substr(0,3).toUpperCase();
					// 1 ,2
					var disk_index = parseInt(disk_type_name.substr(3,1));
					// OSS, MDS
					var node_type_name = $(object).data('node_type_name');
					// 셀렉트 박스 아아디
					var select_id = $(object).data('index') + '-device';
					// /dev/sdb1 , /dev/sdc1 추출
					var device_name = $('#'+select_id).val();
					// 사용자에게 보낼 메세지 생성
					message += '\n\nRestore#'+index+' info\n'+'Disk Device : ' + device_name + '\nHost Name : '+host_name + '\nbackup file : '+ restore_file;
					return {
						host_name : host_name,
						disk_type : disk_type,
						disk_size : $('#'+select_id + " option:selected").attr('size'),
						disk_name : device_name,
						index : disk_index,
						backup_file_localtion : restore_file,
						file_system_num : $('#fs_num').val(),
					}
				}
			)
			
			console.log(send_data)
			// 사용자한테 물어봄
			if(confirm(message)){
				// 사용자가 승인할경우 서버에 전송
				return $.ajax({
					url: contextPath+'/api/v1/lustre/RestoreDisks',
					type: "POST",
					data: JSON.stringify( send_data ),
					// 서버에 보내기전에 processing.. 문구 출력
					beforeSend : function( jqXHR , settings) {
						_.each($('.backup-files:checked'),function(object){
							//var id = $(object).attr('id');
							var index = $(object).data('index');
							var id = index + '-device';
							$('.restore-result[data-target='+id+']').text('processing..');
						});
						$('.backup-files').prop('disabled',true);
						$('.restore-device-name').prop('disabled',true);
					},
					complete : function(jqXHR,textStatus) {
						$('.backup-files').prop('disabled',false);
						$('.restore-device-name').prop('disabled',false);
						_.each($('.backup-files:checked'),function(object){
							//var id = $(object).attr('id');
							var index = $(object).data('index');
							var id = index + '-device';
							$('.restore-result[data-target='+id+']').text('done.');
						});
					},
					dataType: "json",
				    contentType : 'application/json'
				});
			}
		},
		
		
		/**
		 * 이벤트 처리 메서드 리스트
		 */
		listener : function() {
			$('#restore-reset').off('click').on('click',function(e){
				Restore.init();
			})
			$('#restore-apply').off('click').on('click',function(e){
				Restore.restoreDisk().then(function() {
					alert('restore complete');
				})
			})
		}
}