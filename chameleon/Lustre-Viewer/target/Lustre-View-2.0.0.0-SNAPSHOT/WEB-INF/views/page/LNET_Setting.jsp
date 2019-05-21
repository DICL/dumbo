<%@ page contentType="text/html; charset=UTF-8" %>
<!DOCTYPE html>
<html>

<jsp:include page="commons/head.jsp" flush="true"></jsp:include>


<body>
	<div id="root">
		<div class="container">
			<div class="row">
			  	<jsp:include page="commons/header.jsp" flush="true"></jsp:include>
			  	
				<div class="col-12 mt-5">
					<h2>LNET Setting</h2>
					<div id="LNET_Setting_List" class="col-12 border" style="margin-bottom: 70px;">
						<div class="col-12 px-3 py-3">
							<div class="form-group col-12">
						      <div class="form-inline mb-1">
						      	<div class="col-4">
						      		master.hadoop.com MDT device and size
						      	</div>
						      	<div class="col-8">
						      		<button type="button" id="" class="btn btn-primary">Start</button>
									<button type="button" id="" class="btn btn-primary">Stop</button>
						      	</div>
						      </div>
						      <div class="form-inline">
						      	<textarea rows="" cols="" class="form-control col-12"></textarea>
						      </div>
						    </div>
						</div>
						
						<div class="col-12 px-3 py-3">
							<div class="form-group col-12">
						      <div class="form-inline mb-1">
						      	<div class="col-4">
						      		master.hadoop.com MDT device and size
						      	</div>
						      	<div class="col-8">
						      		<button type="button" id="" class="btn btn-primary">Start</button>
									<button type="button" id="" class="btn btn-primary">Stop</button>
						      	</div>
						      </div>
						      <div class="form-inline">
						      	<textarea rows="" cols="" class="form-control col-12"></textarea>
						      </div>
						    </div>
						</div>
						
						<div class="col-12 px-3 py-3">
							<div class="form-group col-12">
						      <div class="form-inline mb-1">
						      	<div class="col-4">
						      		master.hadoop.com MDT device and size
						      	</div>
						      	<div class="col-8">
						      		<button type="button" id="" class="btn btn-primary">Start</button>
									<button type="button" id="" class="btn btn-primary">Stop</button>
						      	</div>
						      </div>
						      <div class="form-inline">
						      	<textarea rows="" cols="" class="form-control col-12"></textarea>
						      </div>
						    </div>
						</div>
						
						<div class="col-12 px-3 py-3">
							<div class="form-group col-12">
						      <div class="form-inline mb-1">
						      	<div class="col-4">
						      		master.hadoop.com MDT device and size
						      	</div>
						      	<div class="col-8">
						      		<button type="button" id="" class="btn btn-primary">Start</button>
									<button type="button" id="" class="btn btn-primary">Stop</button>
						      	</div>
						      </div>
						      <div class="form-inline">
						      	<textarea rows="" cols="" class="form-control col-12"></textarea>
						      </div>
						    </div>
						</div>
						
						<div class="col-12 px-3 py-3 text-right">
							<button type="button" id="mds-apply" class="btn btn-primary">start all</button>
							<button type="button" id="mds-reset" class="btn btn-primary">stop all</button>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</body>


<jsp:include page="commons/javascropts.jsp" flush="true"></jsp:include>
<jsp:include page="commons/modals.jsp" flush="true"></jsp:include>
</html>