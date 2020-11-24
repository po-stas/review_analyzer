$('#contact-form').bootstrapValidator({
//        live: 'disabled',
        message: 'This value is not valid',
        feedbackIcons: {
            valid: 'glyphicon glyphicon-ok',
            invalid: 'glyphicon glyphicon-remove',
            validating: 'glyphicon glyphicon-refresh'
        },
        fields: {
            Review: {
                validators: {
                    notEmpty: {
                        message: 'The Review is required and cannot be empty'
                    }
                }
            }
        }
    });