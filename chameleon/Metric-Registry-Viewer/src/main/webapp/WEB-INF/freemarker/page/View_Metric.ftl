<#ftl encoding="utf-8"/>
<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Metric Registry View</title>
    <link rel="shortcut icon" href="${contextPath}/image/favicon.ico">
    <script src="${contextPath}/npm/jquery/dist/jquery.js"></script>
	<script src="${contextPath}/npm/axios/dist/axios.js"></script>
	
	

	
	<link rel="stylesheet" href="${contextPath}/npm/font-awesome/css/font-awesome.css" >
	<link rel="stylesheet" href="${contextPath}/npm/ionicons/dist/css/ionicons.css" >
	
	
	
	<script src="${contextPath}/npm/bootstrap/dist/js/bootstrap.js"></script>
	<link rel="stylesheet" href="${contextPath}/npm/bootstrap/dist/css/bootstrap.css" >
	
	<script src="${contextPath}/npm/handlebars/dist/handlebars.js"></script>
	
	<script src="${contextPath}/npm/spin/dist/spin.js"></script>
	
	<script src="${contextPath}/npm/underscore/underscore.js"></script>
	
	<link rel="stylesheet" href="${contextPath}/npm/select2/dist/css/select2.min.css" >
	<script src="${contextPath}/npm/select2/dist/js/select2.min.js"></script>
    
    <link rel="stylesheet" href="${contextPath}/css/index.css" >
	<link rel="stylesheet" href="${contextPath}/css/loading.css" >
</head>


<body>
	<div id="root">
		<div class="container">
			<div class="row">
			  	<!-- loading bar -->
				<div class="pure-loarding">
					<div class="lds-spinner">
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
						<div></div>
					</div>
					<p class="message">
						test
					</p>
				</div>
			  	
				<div class="col-12 mt-5">
					<div class="col-12 px-3 py-3 border">
						<h2>Modify metric</h2>
						<div class="col-12 px-3 py-3 ">
							<div class="row mb-3">
								<div class="col-3">
									<label>Name :</label>
								</div>
								<div class="col-9 text-left">
									<input name="num" id="num" type="hidden" class="form-control"  value="${metric.num }"/>
									<input name="name" id="name" type="text" class="form-control"  value="${metric.name }"/>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-3">
									<label>Description :</label>
								</div>
								<div class="col-9 text-left">
									<input name="description" id="description" type="text" class="form-control" value="${metric.description }"/>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-3">
									<label>PID Symbol :</label>
								</div>
								<div class="col-9 text-left">
									<input name="pid_symbol" id="pid_symbol" type="text" class="form-control" value="${metric.pid_symbol }"/>
								</div>
							</div>
							<div class="row mb-3">
								<div class="col-3">
									<label>Y-axis Label :</label>
								</div>
								<div class="col-9 text-left">
									<input name="y_axis_label" id="y_axis_label" type="text" class="form-control" value="${metric.y_axis_label }"/>
								</div>
							</div>
							<div class="row">
								<div class="col-12">
									<label>Parser script:</label>
								</div>
								<div class="col-12">
									<textarea name="parser_script" id="parser_script" rows="15" cols="" class="form-control"  style="width: 100%">${metric.parser_script }</textarea>
								</div>
							</div>
							
						</div>
						<div class="col-12 px-3 py-3 text-right">
							<button type="button" id="update_metric" class="btn btn-primary">Update</button>
							<a href="${contextPath}/" class="btn btn-primary">Cancel</a>
						</div>
					</div>
					
				</div>
			</div>
		</div>
	</div>
</body>

<script>
var contextPath = "${contextPath}";
</script>

<script type="text/javascript" src="${contextPath}/js/common.js"></script>
<script type="text/javascript" src="${contextPath}/js/Handlebars.custom.js"></script>
<script type="text/javascript" src="${contextPath}/js/operationsRunning.js"></script>
<script type="text/javascript" src="${contextPath}/js/page/View_Metric.js"></script>

<div class="modal fade bd-managerque-modal-lg" tabindex="-1" role="dialog" aria-labelledby="myLargeModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-lg">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Job List</h5>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
      </div>
    </div>
  </div>
</div>
</html>