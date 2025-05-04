let layer_num = 0;

$(document).ready( function(){

    drawMaterial();

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
    $("#moveL").on("click", function() {moveLayer(-1);})
    $("#moveR").on("click", function() {moveLayer(+1);})
    $("#update").on("click", function() {updateLayer();})
    $("#destroy").on("click", function() {eel.remove_layer(layer_num-1); drawMaterial();})
    $(".canvas > div").on("click", selectLayer);


    console.log("ready");
  

});

async function drawMaterial() {
    layers = await eel.getLayers()();

    let labels = [];
    layers.reduce(function(dummy, layer) {labels.push(layer.name);}, 0);
    await drawMaterial_core(labels);
    $(".canvas > div").on("click", selectLayer);
};

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
    for (let key in layer) { complete &= layer[key] != ''; }
    
    if (complete) {
        eel.setLayers(layer);
        $("#helpbar").css("color","#ffffff");
        $("#helpbar").text("Layer added correctly");
        $("#insert_panel input").val("");
    } else {
        $("#helpbar").css("color","#ff5555");
        $("#helpbar").text("Cannot add the layer: Some material properties are missing");
    }

    drawMaterial();
}


function selectLayer() {

    layer_num = $(this).index();

    $("#modify_header").text("Modify Layer " + layer_num + " Menu: " + layers[layer_num - 1].name)
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

function moveLayer(move) {
    
    if (layer_num + move > 0 && layer_num + move < layers.length + 1) {  
        eel.move_layer( layer_num - 1, layer_num - 1 + move);
        layer_num += move;
    }

    $("#modify_header").text("Modify Layer " + layer_num + " Menu: " + layers[layer_num - 1].name)
    drawMaterial();
}


function updateLayer() {

    let layer = {
        name:   $("#modify_panel .name_input").val(),
        length: $("#modify_panel .leng_input").val(),
        rho:    $("#modify_panel .dens_input").val(),
        K: [],
        C: [],
        G: []
    };

    layer.K[0] = $("#modify_panel .K_input:eq(0)").val();
    layer.K[1] = $("#modify_panel .K_input:eq(1)").val();
    layer.C[0] = $("#modify_panel .C_input:eq(0)").val();
    layer.C[1] = $("#modify_panel .C_input:eq(1)").val();
    layer.G[0] = $("#modify_panel .G_input:eq(0)").val();

    eel.setLayers(layer, layer_num - 1);
    $("#helpbar").css("color","#ffffff");
    $("#helpbar").text("Layer modified correctly");
    $("#insert_panel input").val("");


    drawMaterial();
}
