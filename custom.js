function showDescriptive(e) {
  for (let i = 0; i < 1 + 1; i++) {
      var x = document.getElementById("descriptive"+i);
      if (e==i) {
        if (x.style.display === "none") {
          x.style.display = "block";
        } else {
          x.style.display = "none";
        }
      } else {
        x.style.display = "none";
      }
    }
  }
  
function showCounterfactual(e) {
  for (let i = 0; i < 1 + 1; i++) {
    var x = document.getElementById("counterfactual"+i);
    if (e==i) {
      if (x.style.display === "none") {
        x.style.display = "block";
      } else {
        x.style.display = "none";
      }
    } else {
      x.style.display = "none";
    }
  }
}
  
function showPlanning(e) {
  for (let i = 0; i < 1 + 1; i++) {
    var x = document.getElementById("planning"+i);
    if (e==i) {
      if (x.style.display === "none") {
        x.style.display = "block";
      } else {
        x.style.display = "none";
      }
    } else {
      x.style.display = "none";
    }
  }
}
 

function showExample(e) {
  for (let i = 0; i < 1 + 1; i++) {
    var x = document.getElementById("example_"+i);
    if (e==i) {
      if (x.style.display === "none") {
        x.style.display = "block";
      } else {
        x.style.display = "none";
      }
    } else {
      x.style.display = "none";
    }
  }
}

function showCounterVideo(e) {
  for (let i = 0; i < 1 + 1; i++) {
    var x = document.getElementById("counterfactualVideo"+i);
    if (e==i) {
      if (x.style.display === "none") {
        x.style.display = "block";
      } else {
        x.style.display = "none";
      }
    } else {
      x.style.display = "none";
    }
  }
}