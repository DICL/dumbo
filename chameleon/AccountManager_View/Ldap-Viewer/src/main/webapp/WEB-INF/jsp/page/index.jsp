<%@ page contentType="text/html; charset=UTF-8" %>
<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LDAP View</title>
	
	
	
    <!--    공통항목-->
    <link href="${contextPath}/resources/bootstrap-3.3.2-dist/css/bootstrap.min.css" rel="stylesheet">
    <!--    index 페이지 전용-->
    <link href="${contextPath}/resources/css/index.css" rel="stylesheet">
    <link href="${contextPath}/resources/node_modules/tui-grid/dist/tui-grid.css" rel="stylesheet">
    <link href="${contextPath}/resources/css/common.css" rel="stylesheet">
</head>

<body>





    <div id="starter-template" class="container theme-showcase">
        <div class="col-lg-12">
            <div class="page-header">
                <h1>사용자 정보</h1>
            </div>
            <div class="col-lg-12">
                <div class="table" id="grid">

                </div>
            </div>
            <div class="col-lg-12" style="text-align: center;">
                <button type="button" class="btn add_user btn-primary" data-toggle="modal" data-backdrop="static"  data-target="#myModal">생성</button>
                <button type="button" class="btn del_user btn-danger">삭제</button>
            </div>
        </div>


    </div>


    <!-- Modal -->
    <div id="myModal" class="modal fade" role="dialog">
        <div class="modal-dialog">

            <!-- Modal content-->
            <div class="modal-content">
                <div class="modal-header bg-primary">
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                    <h4 class="modal-title">사용자 추가</h4>
                </div>
                <form id="addUser">
	                <div class="modal-body" role="form">
                        <div class="form-group">
                            <label>사용자명</label>
                            <input type="text" class="form-control" name="userName">
                        </div>
                        <div class="form-group">
                            <label>그룹</label>
                            <input type="text" class="form-control">
                        </div>
                        <div class="form-group">
                            <label>권한</label>
                            <input type="text" class="form-control">
                        </div>
    	            </div>
    	            <div class="modal-footer">
    	                <button type="submit" class="btn btn-default">확인</button>
    	                <button type="button" class="btn btn-default" data-dismiss="modal">닫기</button>
    	            </div>
                </form>
            </div>
        </div>
    </div>



    <!--    공통항목-->
    <script src="${contextPath}/resources/node_modules/jquery/dist/jquery.js"></script>
	
	<script type="text/javascript">
		//경로설정
		var contextPath = "${contextPath}";
		$.ajaxSetup({
			
		});
	</script>


    <!--    index 페이지 전용-->
    <script src="${contextPath}/resources/node_modules/underscore/underscore.js"></script>
    <script src="${contextPath}/resources/node_modules/backbone/backbone.js"></script>
    <script src="${contextPath}/resources/node_modules/tui-code-snippet/dist/tui-code-snippet.js"></script>
    <script src="${contextPath}/resources/node_modules/tui-pagination/dist/tui-pagination.js"></script>
    <script src="${contextPath}/resources/node_modules/tui-date-picker/dist/tui-date-picker.js"></script>
    <script src="${contextPath}/resources/node_modules/tui-grid/dist/tui-grid.js"></script>

    <!--    공통항목-->
    <script src="${contextPath}/resources/bootstrap-3.3.2-dist/js/bootstrap.min.js"></script>
    <script src="${contextPath}/resources/js/common.js"></script>
    <!--    index 페이지 전용-->
    <script src="${contextPath}/resources/js/index.js"></script>
</body>

</html>