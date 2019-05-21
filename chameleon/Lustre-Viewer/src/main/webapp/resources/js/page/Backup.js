$(function(){
	Backup.init();
});

var Backup = {
		init : function() {
			showLoading('get lustre information')
			Backup.getLustreList(
					$('#fs_num').val()
			).then(
					Backup.gridNodes,
					Backup.errorAjax
			);
		},
		
		/**
		 * 에러시 처리되는 메서드
		 */
		errorAjax : function(jqXHR,textStatus,errorThrown) {
			console.error(jqXHR,textStatus,errorThrown);
		},
		
		/**
		 * Lustre node 정보 가져오기
		 */
		getLustreList : function(file_system_num) {
			return $.ajax({
				url: contextPath+'/api/v1/ambari/getLustreNodes',
				type: "POST",
				data: {
					file_system_num : file_system_num
				}
			})
		},
		
				
		/**
		 * Lustre Node 정보들을 토대로 화면에 그리는 메서드
		 */
		gridNodes : function( data, textStatus, jqXHR) {
			console.log('lustre nodes',data);
			var result = 
				// _.flatten([[1,2,3],[1,4]]) = [1,2,3,4]
				// 리스트 평탄화
				_.flatten(
					// disk list 만 추출
					_.map(
						// 추출할 대상중에서 'MDS'만 제외
						_.filter(
							data,
							function(filter_target) {
								return filter_target.node_type !== 'MDS';
							}	
						),
						// 디스크 리스트만 선택
						function(lustre_node){
							var temp = 
								_.map(
										lustre_node.disk_list,
										// 디스크 리스트에 추가적으로 호스트네임, MDS, OSS , 여부도 추출
										function(disk_info){ 
											disk_info.host_name = lustre_node.host_name;
											disk_info.node_type = lustre_node.node_type;
											disk_info.node_index = lustre_node.index;
											return disk_info; 
										}
									);
							return temp;
						}
					)
				);
			
			console.log(result)
			var target = document.getElementById("backup-page")
			templete_html(target,"Backup",result,function(){
				hideLoading();
				Backup.listener();
			});
			
			
			
		},
		
		send_server : function() {
			var disk_list_text = "";
			
			var send_data = {};
			send_data =
				_.map(
					$('.disk-info:checked'),
					function(object){
						disk_list_text += "\n"+ $(object).data('disk_type') + $(object).data('disk_index')+ ": "+ $(object).val();
						return {
							host_name : $(object).data('host_name'),
							//disk_type : $(object).data('disk_type'),
							disk_type : "OST",
							num : $(object).data('disk_num'),
							disk_name : $(object).val(),
							backup_file_localtion : $('#file_localtion').val(),
							file_system_num : $('#fs_num').val() !== '' ? $('#fs_num').val() : null,
						}
					}
				)
			console.log(send_data)
			if(send_data <= 0 || !confirm("Are you sure backup disk ??" +disk_list_text) ){
				return false;
			}
			
			return $.ajax({
				url: contextPath+'/api/v1/lustre/BackupDisks',
				type: "POST",
				data: JSON.stringify( send_data ),
				beforeSend : function( jqXHR , settings) {
					_.each($('.disk-info:checked'),function(object){
						var id = $(object).attr('id');
						$('.backup-result[data-target='+id+']').text('processing..');
					});
					$('.disk-info').prop('disabled',true);
				},
				complete : function(jqXHR,textStatus) {
					$('.disk-info').prop('disabled',false);
					_.each($('.disk-info:checked'),function(object){
						var id = $(object).attr('id');
						$('.backup-result[data-target='+id+']').text('done.');
					});
				},
				dataType: "json",
			    contentType : 'application/json'
			});
		},
		
		/**
		 * 이벤트 처리 메서드 리스트
		 */
		listener : function() {
			$('#backup-reset').off('click').on('click',function(e){
				Backup.init();
			})
			$('#backup-apply').off('click').on('click',function(e){
				Backup.send_server().then(
					function(data, textStatus, jqXHR) {
						console.log(data)
						alert('backup disk');
					},
					Backup.errorAjax	
				)
			})
		},
}