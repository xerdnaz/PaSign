var selDiv = "";
                
document.addEventListener("DOMContentLoaded", init, false);

function init() {
  document.querySelector('#files').addEventListener('change', handleFileSelect, false);
  selDiv = document.querySelector("#selectedFiles");
}
  
function handleFileSelect(e) {
  if (!e.target.files) return;

  var files = e.target.files;
  for (var i = 0; i < files.length; i++) {
    var f = files[i];

    selDiv.innerHTML += f.name + "<br/>";

    if (f.type.match('image.*')) {
      var reader = new FileReader();

      reader.onload = function (e) {
        var img = document.createElement('img');
        img.src = e.target.result;
        img.style.maxWidth = '100%'; // Set max width to ensure it fits within the container
        document.getElementById('signaturePreview').innerHTML = ''; // Clear previous image
        document.getElementById('signaturePreview').appendChild(img);
      }

      reader.readAsDataURL(f);
    }
  }
}
             
    