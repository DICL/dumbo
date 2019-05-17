// is not use file
$(document).ready(function(){
	setUserGrid();
});

var setUserGrid = function(){
    var url = contextPath + "/ldap/getUserList";
    $.ajax({
        url: url,
        data:{},
        method:"POST",
        success:function(data){
            grid.setData(data.result);
            indexStatus();
        }
    });
}


var grid = new tui.Grid({
    el: $('#grid'),
    scrollX: false,
    scrollY: false,
    rowHeaders: [
    	{
            title: '',
			type: 'radio',
		}
	],
    columns: [
        {
            title: '사용자명',
            name: 'userName'
        },
        {
            title: '권한',
            name: 'userPermission'
        },
        {
            title: '그룹',
            name: 'userGroup'
        },
        {
            title: '생성일자',
            name: 'creatDate'
        },
        
    ]
});


var addUser = function(data) {
	if(!data) return;
	var url = contextPath + "/ldap/setUser";
    $.ajax({
        url: url,
        data:data,
        method:"POST",
        success:function(data){
        	$("#myModal").modal('hide');
        	$('#addUser')[0].reset();
        	alert("등록 되었습니다.");
        	setUserGrid();
        }
    });
}

var delUser = function() {
	if(!confirm("정말로 삭제하시겠습니까?")){
		return;
	}
	var data = grid.getCheckedRows()[0];
	var url = contextPath + "/ldap/delUser";
	$.ajax({
        url: url,
        data: data,
        method:"POST",
        success:function(data){
        	alert("삭제되었습니다.");
        	setUserGrid();
        }
    });
}



var indexStatus = function(){
	$("#addUser").off('submit').on('submit',function(){
		addUser({
			"userName" : $("#myModal input[name=userName]").val(),
		});
		return false;
	});
	$(".del_user").off('click').on('click',function(){
		delUser();
	});
}