<#ftl encoding="utf-8"/>
<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>AccountManager View</title>
	
	
	
    <!--    공통항목(부트스트랩) -->
    <link href="${contextPath}/resources/node_modules/bootstrap/dist/css/bootstrap.min.css" rel="stylesheet">
    <!--    index 페이지 전용-->
    <link href="${contextPath}/resources/css/index2.css" rel="stylesheet">
    <link href="${contextPath}/resources/css/common.css" rel="stylesheet">
    <!--    jstree-->
    <link href="${contextPath}/resources/node_modules/jstree-bootstrap-theme/dist/themes/proton/style.min.css" rel="stylesheet">
</head>

<body>




 
    <div class="container theme-showcase">
        <div class="col-md-12">
            <div class="page-header" style="text-align: center;">
                <h1>AccountManager View</h1>
            </div>
            <div class="row">
            	<div class="col-md-12" >
                    <h5 id="serverInfo">&nbsp;</h5>
                </div>
                <div class="col-md-4">
                    <div class="table" id="tree">
                    
                    </div>
                </div>
                <div class="col-md-8" id="content-body">
                    <div class="panel panel-default" >
                      <div class="panel-heading">&nbsp;</div>
                      <div class="panel-body">
                           
                      </div>
                      <div class="panel-footer">

                      </div>
                    </div>
                </div>
               
            </div>
            <div class="col-md-12 btn-container">
                <button type="button" class="btn add_user btn-default">Add User</button>
                <!-- <button type="button" class="btn create_home_folder btn-default">Create WorkingDir</button> -->
            </div>
        </div>
    </div>


    <!-- Modal -->
    <div id="myModal" class="modal fade" role="dialog">
            <div class="modal-dialog">
                <!-- Modal content-->
                <form role="form" id="addUser">
                    <div class="modal-content">
                        <div class="modal-header bg-primary">
                            <button type="button" class="close" data-dismiss="modal">&times;</button>
                            <h4 class="modal-title">Add User</h4>
                        </div>
                        <div class="modal-body">
                            
                            <div class="form-group row">
                                <label for="sn" class="col-sm-3 col-form-label">sn</label>
                                <div class="col-sm-9">
                                    <input type="text" class="form-control" id="sn" name="sn" />
                                </div>
                            </div>
                            <div class="form-group row">
                                <label for="cn" class="col-sm-3 col-form-label">cn</label>
                                <div class="col-sm-9">
                                    <input type="text" class="form-control" id="cn" name="cn" />
                                </div>
                            </div>
                            <div class="form-group row">
                                <label for="uid" class="col-sm-3 col-form-label">uid</label>
                                <div class="col-sm-9">
                                    <input type="text" class="form-control" id="uid" name="uid" />
                                </div>
                            </div>
                            <div class="form-group row">
                                <label for="userpassword" class="col-sm-3 col-form-label">userPassword</label>
                                <div class="col-sm-9">
                                    <input type="password" class="form-control" id="userpassword" name="userPassword" />
                                </div>
                            </div>
                            
                            <div class="form-group row">
                                <label for="uidNumber" class="col-sm-3 col-form-label">uidNumber</label>
                                <div class="col-sm-9">
                                    <input type="text" class="form-control" id="uidNumber" name="uidNumber" />
                                </div>
                            </div>
                            <div class="form-group row">
                                <label for="gidNumber" class="col-sm-3 col-form-label">gidNumber</label>
                                <div class="col-sm-9">
                                    <input type="text" class="form-control" id="gidNumber" name="gidNumber" />
                                </div>
                            </div>
                            <div class="form-group row">
                                <label for="homeDirectory" class="col-sm-3 col-form-label">homeDirectory</label>
                                <div class="col-sm-9">
                                    <input type="text" class="form-control" id="homeDirectory" name="homeDirectory" readonly="readonly" />
                                </div>
                            </div>
                            <div class="form-group row">
                                <label for="loginShell" class="col-sm-3 col-form-label">loginShell</label>
                                <div class="col-sm-9">
                                    <input type="text" class="form-control" id="loginShell" name="loginShell" readonly="readonly" value="/bin/bash"/>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn_add_user btn-default">Confirm</button>
                            <button type="button" class="btn btn-default" data-dismiss="modal">Close</button>
                        </div>
                    </div>
                </form>
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
	<!--    hadlebars js-->
    <script src="${contextPath}/resources/node_modules/handlebars/dist/handlebars.min.js"></script>
    <!--  	spin.js-->
    <script src="${contextPath}/resources/node_modules/spin/dist/spin.min.js"></script>
    <!--    공통항목(jstrre)-->
    <script src="${contextPath}/resources/node_modules/jstree/dist/jstree.min.js"></script>
    <!--    공통항목(부트스트랩)-->
    <script src="${contextPath}/resources/node_modules/bootstrap/dist/js/bootstrap.min.js"></script>
    <script src="${contextPath}/resources/js/common.js"></script>
    <!--    index 페이지 전용-->
    <script src="${contextPath}/resources/js/index2.js"></script>
</body>

</html>