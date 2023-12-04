let layer_num = 0;

function setListener() {$(".canvas > div").on("click", selectLayer);}

function addLayer() {
    console.log("Adding a new layer: " +  $("#insert_panel #name_input").val());

    let layer = {
        name:   $("#insert_panel .name_input").val(),
        length: $("#insert_panel .leng_input").val(),
        rho:    $("#insert_panel .dens_input").val(),
        K: [],
        C: [],
        G: []
    };

    layer.K[0] = $("#insert_panel .K_input:eq(0)").val();
    layer.K[1] = $("#insert_panel .K_input:eq(1)").val();
    layer.C[0] = $("#insert_panel .C_input:eq(0)").val();
    layer.C[1] = $("#insert_panel .C_input:eq(1)").val();
    layer.G[0] = $("#insert_panel .G_input:eq(0)").val();
    
    let complete = true;
    for (let key in layer) {
        complete &= layer[key] != ''
    }
    if (complete) {
        eel.setLayer(layer);
        $("#helpbar").css("color","#ffffff");
        $("#helpbar").text("Layer added correctly");
        $("#insert_panel input").val("");
    } else {
        $("#helpbar").css("color","#ff5555");
        $("#helpbar").text("Cannot add the layer: Some material properties are missing");
    }

    drawMaterial(setListener);
}

function selectLayer() {

    layer_num = $(this).index();

    $("#modify_header").text("Modify Layer " + layer_num + " Menu")
    $("#modify_panel").slideDown(300);
    $("#insert_panel").slideUp(300);

    $("#modify_panel .name_input").val(layers[layer_num - 1].name  )
    $("#modify_panel .leng_input").val(layers[layer_num - 1].length)
    $("#modify_panel .dens_input").val(layers[layer_num - 1].rho   )

    $("#modify_panel .K_input:eq(0)").val(layers[layer_num - 1].K[0])
    $("#modify_panel .K_input:eq(1)").val(layers[layer_num - 1].K[1])
    $("#modify_panel .C_input:eq(0)").val(layers[layer_num - 1].C[0])
    $("#modify_panel .C_input:eq(1)").val(layers[layer_num - 1].C[1])
    $("#modify_panel .G_input:eq(0)").val(layers[layer_num - 1].G[0])

}



$(document).ready( function(){

    drawMaterial(setListener);

    $("#insert_header").on("click", function() {
        $("#modify_panel").slideUp(300);
        $("#insert_panel").slideToggle(300);
    });

    $("#modify_header").on("click", function() {
        if (layer_num > 0) {
            $("#insert_panel").slideUp(300);
            $("#modify_panel").slideToggle(300);    
        }
    });

    $("#insert_panel #submit").on("click", addLayer);
    $(".canvas > div").on("click", selectLayer);


    console.log("ready");
  

});
