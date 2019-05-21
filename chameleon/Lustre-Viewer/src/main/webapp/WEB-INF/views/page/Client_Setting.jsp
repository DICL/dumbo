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
			  		<h2>Client Setting</h2>
			  		<div class="col-12 border">
			  			<div class="col-12 px-3 py-3">
			  				<div id="client_page" class="form-group col-12 mb-3">
			  				</div>
			  				<!-- <div class="form-group col-12 mb-3">
						      <label for="mdt">
						      	 Client IO Network
						      </label>
						      <div class="form-inline">
						      	<input type="text" id="client-network-name" class="form-control col-6" value="enp0s8"/>
						      </div>
						    </div> -->
						    
						    <div class="form-group col-12 mb-3">
						      <label for="mdt">
						      	 Client Mount Point
						      </label>
						      <div class="form-inline">
						      	<input type="text" id="client-mount-name" class="form-control col-6" value="/lustre"/>
						      </div>
						    </div>
			  			</div>
			  			<div class="col-12 px-3 py-3 text-right">
							<button id="client-apply" type="button" class="btn btn-primary">Apply</button>
							<button id="client-reset" type="button" class="btn btn-primary">Reset</button>
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