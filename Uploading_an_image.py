{% extends "public/templates/public_template.html"%}

{% block title%}Upload image{%endblock%}

(% block main %)

<div class= "container">
    <div class="row">
        <div class = "col">


            <h1>Upload image</h1>
            <hr>

            <form action = " /upload -image" method= "POST" enctype= "/multipart/form-data">

                <div class="form group">
                    <label> Seclect image</label>
                    <div class="custom-file">
                        <input type="file" class="custom-file-input" name="image" id="image">
                        <label class="custom-file-label" for="image">Select image...</label>
                    </div>
                </div>
        
                <button type="submit" class="btn btn-primary">Upload</button>

            
                
            </form>       
        
        </div> 
    </div>   
</div>

{% endblock %}

